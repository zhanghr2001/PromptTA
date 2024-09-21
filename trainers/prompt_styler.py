from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import TensorDataset, DataLoader

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import mkdir_if_missing
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers.base_sfdg import *
from utils.clip_part import *
from utils.loss import AngularPenaltySMLoss


class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PROMPT_STYLER.N_CTX
        ctx_init = cfg.TRAINER.PROMPT_STYLER.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]   # text encoder hidden size(512)
        self.dim = clip_model.text_projection.shape[1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        self.ctx_vectors = []
        for _ in range(cfg.TRAINER.PROMPT_STYLER.K_PROPMT):
            ctx_vector = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vector, std=0.02)
            ctx_vector = nn.Parameter(ctx_vector)
            self.ctx_vectors.append(ctx_vector)
        self.ctx_vectors = nn.ParameterList(self.ctx_vectors)
    
        ctx_init = ctx_init.replace("_", " ")

        print('Prompt design: Prompt-driven Style Generation')
        print(f'Initial context: "{ctx_init}"')
        print(f"Number of Prompt Styler context words (tokens): {n_ctx}")
        
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [ctx_init + " " + name + "." for name in classnames]

        tokenized_ctx_init = clip.tokenize(ctx_init)
        tokenized_classnames = torch.cat([clip.tokenize(classname) for classname in classnames])  # (n_cls, n_tkn)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            ctx_init_embedding = clip_model.token_embedding(tokenized_ctx_init).type(dtype)
            self.classname_embedding = clip_model.token_embedding(tokenized_classnames).type(dtype)
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # 0   1 2 3     4  5 6     7 8~76
        # sos a X style of a [cls] . EOS
        self.register_buffer("token_prefix", embedding[:, :2, :])  # SOS + "a"
        self.register_buffer("token_suffix", embedding[:, 2 + n_ctx:, :])  # CLS + EOS ("of a [cls].")
        self.register_buffer("token_prefix_init", ctx_init_embedding[:, :2, :])  # SOS + "a"
        self.register_buffer("token_suffix_init", ctx_init_embedding[:, 2 + n_ctx:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_ctx_init = tokenized_ctx_init
        self.tokenized_classnames = tokenized_classnames
        self.tokenized_prompts = tokenized_prompts  
        self.classnames = classnames  
        self.class_token_position = cfg.TRAINER.PROMPT_STYLER.CLASS_TOKEN_POSITION

    def forward(self, index=None, style=False):
        ctx = self.ctx_vectors
        if style:
            prefix = self.token_prefix_init
            suffix = self.token_suffix_init
        else:
            prefix = self.token_prefix
            suffix = self.token_suffix
        
        if index == None:  # get all style prompts with classnames (N classes)
            prompts = []
            for ctx_i in ctx:
                if ctx_i.dim() == 2:
                    ctx_i = ctx_i.unsqueeze(0).expand(self.n_cls, -1, -1)
                prompt = self.construct_prompts(ctx_i, prefix, suffix)
                prompts.append(prompt)
            return prompts
        else:
            if style:   # get one style prompt without classnames
                prompt = self.construct_prompts(ctx[index].unsqueeze(0), prefix, suffix)
                return prompt
            else:   # get one style prompt with classnames (N classes)
                prompt = self.construct_prompts(ctx[index].unsqueeze(0).expand(self.n_cls, -1, -1), prefix, suffix)
                return prompt, self.classname_embedding
    

class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.K_PROPMT = cfg.TRAINER.PROMPT_STYLER.K_PROPMT
        
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        self.image_encoder = clip_model.visual

        self.classifier = nn.Linear(cfg.FEAT_DIM, self.prompt_learner.n_cls, bias=False)
        if cfg.TRAINER.PROMPT_STYLER.PREC == "fp16":
            self.classifier.half()

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def image_forward(self, image):   # inference
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        output = image_features
      
        return output
    
    def style_generation(self, i):  # i-th style prompt
        if i == 0:
            L_style = torch.tensor(0.)
        else:
            prompt_style_i = self.prompt_learner(i, True)   # i-th style prompt w/o classnames, i.e., P^style_i
            ctx_feature_i = self.text_encoder(prompt_style_i, self.prompt_learner.tokenized_ctx_init)  
            ctx_feature_i = ctx_feature_i / ctx_feature_i.norm(dim=-1, keepdim=True)

            logits_style_sum = 0
            for j in range(self.K_PROPMT):
                if j >= i:
                    break
                prompt_style_j = self.prompt_learner(j, True)   # j-th style prompt w/o classnames, i.e., P^style_j
                ctx_feature_j = self.text_encoder(prompt_style_j, self.prompt_learner.tokenized_ctx_init) 
                ctx_feature_j = ctx_feature_j / ctx_feature_j.norm(dim=-1, keepdim=True)
                logits = ctx_feature_i @ ctx_feature_j.t() 
                logits_style_sum += torch.abs(logits)

            L_style = logits_style_sum / i

        # prompt_i: [N, 77, feat_dim], i-th style prompt with classnames, i.e., P^style_i + P^content
        # classnames_embedding: [N, 77, feat_dim], all classnames, i.e., P^content
        prompt_i, classnames_embedding = self.prompt_learner(i)     

        tokenized_classname = self.prompt_learner.tokenized_classnames.to(prompt_i.device)
        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(prompt_i.device)

        classnames_features = self.text_encoder(classnames_embedding.to(prompt_i.device), tokenized_classname)  # [N, feat_dim]
        classnames_features = classnames_features / classnames_features.norm(dim=-1, keepdim=True)

        prompt_features_i = self.text_encoder(prompt_i, tokenized_prompts)  # [N, feat_dim]
        prompt_features_i = prompt_features_i / prompt_features_i.norm(dim=-1, keepdim=True)
        
        logits_content = prompt_features_i @ classnames_features.t()    # [N, N]
        content_label = torch.arange(self.prompt_learner.n_cls).to(prompt_i.device)
        L_content = F.cross_entropy(logits_content, content_label)

        return L_style, L_content

    def text_forward(self, input, tokenized_input):
        text_features = self.text_encoder(input, tokenized_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        output = text_features

        return output
    

@TRAINER_REGISTRY.register()
class PROMPT_STYLER(Base_SFDG):
    """Reproduced Prompt Styler.

    PromptStyler: Prompt-driven Style Generation for Source-free Domain Generalization
    https://arxiv.org/abs/2307.15199
    """  
    def build_model(self):
        cfg = self.cfg

        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)
        self.target_domains = cfg.DATASET.TARGET_DOMAINS
        self.save = cfg.SAVE_MODEL
        self.best_test_result = 0.

        self.prompt_start_epoch = self.prompt_epoch = 0
        self.prompt_max_epoch = cfg.OPTIM_PROMPT.MAX_EPOCH

        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM_CLASSIFIER.MAX_EPOCH
        
        if torch.cuda.is_available() and cfg.USE_CUDA:
            if len(cfg.GPU) == 1:
                self.device = torch.device("cuda:{}".format(cfg.GPU))
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPT_STYLER.PREC == "fp32" or cfg.TRAINER.PROMPT_STYLER.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        
        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name or "classifier" in name:
                param.requires_grad_(True)

        print("# Total params: {:,} (prompt: {:,}, classifier: {:,})".format(
            count_num_param(self.model.prompt_learner) + count_num_param(self.model.classifier), 
            count_num_param(self.model.prompt_learner), count_num_param(self.model.classifier)))

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"parameters to be updated: {sorted(enabled)}\n")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(
                self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.ce = False
        self.arcface_loss = AngularPenaltySMLoss(cfg.FEAT_DIM, self.n_cls, loss_type='arcface', s=5, m=0.5)

        # NOTE: only give prompt_learner to the optimizer
        self.optim_prompt = build_optimizer(self.model.prompt_learner, cfg.OPTIM_PROMPT)
        self.sched_prompt = build_lr_scheduler(self.optim_prompt, cfg.OPTIM_PROMPT)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim_prompt, self.sched_prompt)

        self.optim_classifier = build_optimizer(self.model.classifier, cfg.OPTIM_CLASSIFIER)
        self.sched_classifier = build_lr_scheduler(self.optim_classifier, cfg.OPTIM_CLASSIFIER)
        self.register_model("classifier", self.model.classifier, self.optim_classifier, self.sched_classifier)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPT_STYLER.PREC == "amp" else None
    
    def train(self):
        self.before_train()

        # Two-stage training
        self.train_fc(self.get_data())

        self.after_train()
    
    def get_data(self):
        backbone_name = self.cfg.MODEL.BACKBONE.NAME.replace('/', '-')
        data_path = os.path.join('data',  f'{self.cfg.DATASET.NAME}_{backbone_name}_data.pt')
        if not os.path.exists(data_path):
            data = self.style_generation(data_path)
        else:
            data = torch.load(data_path, map_location=self.device)

        return data
        
    def style_generation(self, data_path):
        def reset_learning_rate(optimizer, new_lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        print(f'Generating {self.cfg.TRAINER.PROMPT_STYLER.K_PROPMT} style...')
        for i in range(self.cfg.TRAINER.PROMPT_STYLER.K_PROPMT):
            print(f"Only update {i}-th prompt...")
            for name, param in self.model.prompt_learner.named_parameters():
                param.requires_grad_(False)
                if f"ctx_vectors.{i}" == name:
                    param.requires_grad_(True)

            # Double check
            enabled = set()
            for name, param in self.model.prompt_learner.named_parameters():
                if param.requires_grad:
                    enabled.add(name)
            print(f"parameters to be updated: {sorted(enabled)}")

            for self.prompt_epoch in range(self.prompt_max_epoch):
                L_style, L_content = self.model.style_generation(i)
                L_prompt = L_style + L_content
                
                self.optim_prompt.zero_grad()
                L_prompt.backward()
                self.optim_prompt.step()

                self.sched_prompt.step()

                if (self.prompt_epoch + 1) % 20 == 0:
                    info = []
                    info += [f"prompt [{i+1}/{self.cfg.TRAINER.PROMPT_STYLER.K_PROPMT}]"]
                    info += [f"epoch [{self.prompt_epoch + 1}/{self.prompt_max_epoch}]"]
                    info += [f"loss_prompt {L_prompt.item():.3f}"]
                    info += [f"loss_style {L_style.item():.3f}"]
                    info += [f"loss_content {L_content.item():.3f}"]
                    info += [f"lr {self.get_current_lr(self.optim_prompt):.4e}"]
                    print("\t".join(info))

            reset_learning_rate(self.optim_prompt, self.cfg.OPTIM_PROMPT.LR)
            self.sched_prompt.last_epoch = 0

        data = self.save_style_prompt(data_path)

        return data

    def save_style_prompt(self, data_path):
        print("Turning off gradients in prompt learner...")
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name:
                param.requires_grad_(False)
        
        prompts = self.model.prompt_learner()   # KN prompts with classnames

        prompts_list = []
        tokenized_prompts_list = []
        labels_list = []

        labels = torch.arange(self.n_cls)
        for prompt in prompts:
            prompts_list.append(prompt)
            tokenized_prompts_list.append(self.model.prompt_learner.tokenized_prompts)
            labels_list.append(labels)

        all_prompts = torch.cat(prompts_list, dim=0)
        all_tokenized_prompts = torch.cat(tokenized_prompts_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        data = {
            'prompts': all_prompts,
            'tokenized_prompts': all_tokenized_prompts,
            'labels': all_labels
        }

        torch.save(data, data_path)

        return data

    def train_fc(self, data):
        prompts = data['prompts']
        tokenized_prompts = data['tokenized_prompts']
        labels = data['labels']
        
        # segment to prevent cuda OOM on domainnet
        if "DomainNet" in self.cfg.DATASET.NAME:
            seg=5
            prompts_segs = torch.chunk(prompts, seg, dim=0)
            tokenized_prompts_segs = torch.chunk(tokenized_prompts, seg, dim=0)

            features = []
            with torch.no_grad():
                for i in range(seg):
                    features.append(self.model.text_forward(prompts_segs[i].to(self.device), tokenized_prompts_segs[i].to(self.device)))
                text_features = torch.cat(features, dim = 0)
        else:
            with torch.no_grad():
                text_features = self.model.text_forward(prompts.to(self.device), tokenized_prompts.to(self.device))          

        dataset = TensorDataset(text_features, labels)
        self.train_loader = DataLoader(dataset, batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(dataset, batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE, shuffle=False)

        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()

    def get_current_lr(self, optim):
            return optim.param_groups[0]["lr"]
           
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        len_train_loader = len(self.train_loader)
        self.num_batches = len_train_loader
        train_loader_iter = iter(self.train_loader)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            batch = next(train_loader_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr(self.optim_classifier):.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(self.optim_classifier), n_iter)

            end = time.time()

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.PROMPT_STYLER.PREC
        if prec == "amp":
            with autocast():
                output = input
                output = self.model.classifier(output)
                loss_class = F.cross_entropy(output, label)
            self.optim_classifier.zero_grad()
            self.scaler.scale(loss_class).backward()
            self.scaler.step(self.optim_classifier)
            self.scaler.update()
        else:
            output = input
            output = self.model.classifier(output)
            loss_class = F.cross_entropy(output, label)

            self.optim_classifier.zero_grad()
            loss_class.backward()
            self.optim_classifier.step()

        loss_summary = {
            "loss": loss_class.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.sched_classifier.step()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch[0]        
        label = batch[1]
        
        input = input.to(self.device)
        label = label.to(self.device)
    
        return input, label

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = self.cfg.TEST.DO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 
                                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            curr_test_result = self.test('test')

            is_test_best = curr_test_result > self.best_test_result
            if is_test_best:
                self.best_test_result = curr_test_result
                self.best_test_epoch = self.epoch
                if self.cfg.SAVE_MODEL:
                    self.save_model(self.epoch, self.output_dir, model_name="model-best-test.pth.tar")

            print('******* Domain {} best test acc:     {:.1f}%, epoch: {} *******'.format(self.cfg.TARGET_DOMAIN, self.best_test_result, self.best_test_epoch+1))
            
        self.set_model_mode("train")
        if self.cfg.SAVE_MODEL and (meet_checkpoint_freq or last_epoch):
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print(f"Evaluate on the *val* set")
            for batch_idx, batch in enumerate(data_loader):
                input, tokenized_input, label = self.parse_batch_train(batch)
                output = self.model.text_forward(input, tokenized_input)
                if self.ce:
                    output = self.model.classifier(output)
                else:
                    _, output = self.arcface_loss(self.model.classifier, output, label, return_logits=True)
                self.evaluator.process(output, label)
        else:
            split = "test"
            data_loader = self.test_loader
            print(f"Evaluate on the *test* set")
            for batch_idx, batch in enumerate(data_loader):
                input, label = self.parse_batch_test(batch)
                output = self.model.image_forward(input)
                output = self.model.classifier(output)

                # sim = output @ self.text_features.T
                # sim = (sim * 100).softmax(dim=-1)
                # prefix_embeddings = sim @ self.text_features
                # prefix_embeddings /= prefix_embeddings.norm(dim=-1,keepdim=True)
                # output = self.model.classifier(prefix_embeddings)

                self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return results['accuracy']

