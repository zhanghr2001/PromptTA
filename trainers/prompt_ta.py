import torch
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from trainers.base_sfdg import *
from utils.clip_part import *

from torch.utils.data import *

import numpy as np
from openTSNE import TSNE
from utils import tSNE_utils as utils

class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PROMPT_TA.N_CTX
        ctx_init = cfg.TRAINER.PROMPT_TA.CTX_INIT    # 'A S style of a'
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]   # text encoder hidden size(512)
        self.dim = clip_model.text_projection.shape[1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        self.ctx_vectors = []
        for _ in range(cfg.TRAINER.PROMPT_TA.K_PROPMT):
            ctx_vector = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vector, std=0.02)
            ctx_vector = nn.Parameter(ctx_vector)
            self.ctx_vectors.append(ctx_vector)
        self.ctx_vectors = nn.ParameterList(self.ctx_vectors)   # (k_prompt, n_ctx, ctx_dim)
    
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
        self.register_buffer("token_suffix", embedding[:, 2 + n_ctx:, :])  # CLS + EOS ("of a [cls].")  # "style of a [cls] . EOS"
        self.register_buffer("token_prefix_init", ctx_init_embedding[:, :2, :])  # SOS + "a"
        self.register_buffer("token_suffix_init", ctx_init_embedding[:, 2 + n_ctx:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_ctx_init = tokenized_ctx_init
        self.tokenized_classnames = tokenized_classnames
        self.tokenized_prompts = tokenized_prompts  
        self.classnames = classnames  
        self.class_token_position = cfg.TRAINER.PROMPT_TA.CLASS_TOKEN_POSITION

        domain_bank = ['', 'photo', 'art', 'cartoon', 'sketch', 'clipart', 'infograph', 'painting', 'quickdraw', 'real', 'product']
        text_adpter = self.get_adapter(classnames, domain_bank) # a [domain_bank] photo of a [classnames]
        self.tokenized_adapter = clip.tokenize(text_adpter)
        with torch.no_grad():
            self.adapter_embedding = clip_model.token_embedding(self.tokenized_adapter).type(dtype)
        self.n_domain = len(domain_bank)

    def get_adapter(self, classnames, domain_bank):
        text_classname_list = [[a] for a in classnames]
        text_domain_list = [[a] for a in domain_bank]

        text_all = []

        for t in text_classname_list:
            for s in text_domain_list:
                text_all.append([t,s])
        
        return [f"a {s[0]} photo of a {t[0]}" for t,s in text_all]
    
    def forward(self, index=None, style=False, ctx_vectors=None):       #
        if ctx_vectors == None:
            ctx = self.ctx_vectors
        else:
            ctx = ctx_vectors.unsqueeze(1)
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
                prompt = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, *, dim)
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
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.K_PROPMT = cfg.TRAINER.PROMPT_TA.K_PROPMT
        
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        self.image_encoder = clip_model.visual

        self.classifier = nn.Linear(cfg.FEAT_DIM, self.prompt_learner.n_cls, bias=False)
        if cfg.TRAINER.PROMPT_TA.PREC == "fp16":
            self.classifier.half()

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
    
    def image_forward(self, image):   # image encoder inference
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

    def text_forward(self, input, tokenized_input):     # text encoder inference
        text_features = self.text_encoder(input, tokenized_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        output = text_features

        return output

@TRAINER_REGISTRY.register()
class PROMPT_TA(Base_SFDG):
    """PS + Adapter.
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

        if cfg.TRAINER.PROMPT_TA.PREC == "fp32":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        
        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name or "classifier" in name:
                param.requires_grad_(True)

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
        self.build_cache_model()

        print("# Total params: {:,} (prompt: {:,}, classifier: {:,}, adapter: {:,})".format(
            count_num_param(self.model.prompt_learner) + count_num_param(self.model.classifier) + count_num_param(self.adapter), 
            count_num_param(self.model.prompt_learner), count_num_param(self.model.classifier), count_num_param(self.adapter)
        ))

        self.optim_prompt = build_optimizer(self.model.prompt_learner, cfg.OPTIM_PROMPT)
        self.sched_prompt = build_lr_scheduler(self.optim_prompt, cfg.OPTIM_PROMPT)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim_prompt, self.sched_prompt)

        self.optim_classifier = build_optimizer(self.model.classifier, cfg.OPTIM_CLASSIFIER)
        self.sched_classifier = build_lr_scheduler(self.optim_classifier, cfg.OPTIM_CLASSIFIER)
        self.register_model("classifier", self.model.classifier, self.optim_classifier, self.sched_classifier)

        self.optim_adapter = build_optimizer(self.adapter, cfg.OPTIM_ADAPTER)
        self.sched_adapter = build_lr_scheduler(self.optim_adapter, cfg.OPTIM_ADAPTER)
        self.register_model("adapter", self.adapter, self.optim_adapter, self.sched_adapter)


    def build_cache_model(self):
        with torch.no_grad():
            self.cache_key = self.model.text_encoder(   # a [domain_bank] photo of a [classnames]
                self.model.prompt_learner.adapter_embedding.to(self.device), 
                self.model.prompt_learner.tokenized_adapter.to(self.device))
            self.cache_key = torch.mean(self.cache_key.view(self.n_cls, self.model.prompt_learner.n_domain, -1), dim=1) # n_cls*1024
            self.cache_key = self.cache_key / self.cache_key.norm(dim=-1, keepdim=True)

            self.cache_value = torch.arange(self.n_cls)
            self.cache_value = F.one_hot(self.cache_value).to(self.model.dtype).to(self.device) #  n_cls*n_cls

            self.adapter = nn.Linear(self.cache_key.shape[1], self.cache_key.shape[0], bias=False).to(self.model.dtype)
            self.adapter.weight = nn.Parameter(self.cache_key)
            
        self.alpha = self.cfg.ALPHA
        self.beta = self.cfg.BETA

    def train(self):
        self.before_train()

        # Two-stage training
        self.train_fc(self.get_data())

        self.after_train()

    def train_fc(self, data):
        prompts = data['prompts']   # embedding of "a [ctx_vectors] style of a [classnames]"
        tokenized_prompts = data['tokenized_prompts']
        labels = data['labels']

        # epoch-wise hybrid
        means = torch.zeros((self.n_cls, self.cfg.FEAT_DIM)).to(self.device)
        stds = torch.zeros((self.n_cls, self.cfg.FEAT_DIM)).to(self.device)

        prompt_text_features = []
        with torch.no_grad():
            for cls in range(self.n_cls):
                text_features = self.model.text_forward(prompts[labels == cls], tokenized_prompts[labels == cls])   # text features of current class
                means[cls] = text_features.mean(dim=0)
                stds[cls] = text_features.std(dim=0)
                prompt_text_features.append(text_features)
            prompt_text_features = torch.cat(prompt_text_features, dim=0)

        labels = []
        for cls in range(self.n_cls):
            labels.append(torch.full((self.cfg.TRAINER.PROMPT_TA.K_PROPMT,), cls))
        labels = torch.cat(labels, dim=0)


        for self.epoch in range(self.start_epoch, self.max_epoch):
            if self.epoch % 2 == 0:
                dataset = TensorDataset(prompt_text_features, labels)
            else:
                new_text_features = []
                for cls in range(self.n_cls):
                    # normal             
                    epsilon = torch.randn(self.cfg.TRAINER.PROMPT_TA.K_PROPMT, self.cfg.FEAT_DIM).to(self.device) # resample in every epoch

                    # uniform
                    # epsilon = torch.rand(self.cfg.TRAINER.PROMPT_TA.K_PROPMT, self.cfg.FEAT_DIM).to(self.device)
                    # epsilon = (epsilon - 0.5) * 2

                    new_features = means[cls] + stds[cls] * epsilon
                    new_text_features.append(new_features.half())

                new_text_features = torch.cat(new_text_features, dim=0)
                dataset = TensorDataset(new_text_features, labels)

            self.train_loader = DataLoader(dataset, batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE, shuffle=True)
            # self.val_loader = DataLoader(dataset, batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE, shuffle=False)

            self.before_epoch()
            self.run_epoch()
            self.after_epoch()

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
            loss_summary = self.forward_backward(batch)     # run
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
                if hasattr(self, 'optim_classifier'):
                    info += [f"fc_lr {self.get_current_lr(self.optim_classifier):.4e}"]
                if hasattr(self, 'optim_adapter'):
                    info += [f"ad_lr {self.get_current_lr(self.optim_adapter):.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
                
            if hasattr(self, 'optim_classifier'):
                self.write_scalar("train/fc_lr", self.get_current_lr(self.optim_classifier), n_iter)
            if hasattr(self, 'optim_adapter'):
                self.write_scalar("train/ad_lr", self.get_current_lr(self.optim_adapter), n_iter)

            end = time.time()

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

            print('******* Domain {} best test acc: {:.1f}%, epoch: {} *******'.format(self.cfg.TARGET_DOMAIN, self.best_test_result, self.best_test_epoch+1))
            
        self.set_model_mode("train")
        if self.cfg.SAVE_MODEL and (meet_checkpoint_freq or last_epoch):
            self.save_model(self.epoch, self.output_dir)

    def parse_batch_train(self, batch):
        input = batch[0]        
        label = batch[1]
        
        input = input.to(self.device)
        label = label.to(self.device)
    
        return input, label

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        output = input

        affinity = self.adapter(output)
        cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_value * self.alpha
        loss_cache = F.cross_entropy(cache_logits, label)

        class_logits = self.model.classifier(output)
        loss_class = F.cross_entropy(class_logits, label)
        
        self.optim_adapter.zero_grad()
        loss_cache.backward()
        self.optim_adapter.step()
        
        self.optim_classifier.zero_grad()
        loss_class.backward()
        self.optim_classifier.step()

        loss_summary = {
            "loss": (loss_class+loss_cache).item(),
            "loss_class": loss_class.item(),
            "loss_cache": loss_cache.item(),
            "acc": compute_accuracy(class_logits + self.alpha * cache_logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.sched_classifier.step()
            self.sched_adapter.step()

        return loss_summary

    def get_data(self):     # load or generate style word vectors
        backbone_name = self.cfg.MODEL.BACKBONE.NAME.replace('/', '-')
        data_path = os.path.join('data', f'{self.cfg.DATASET.NAME}_{backbone_name}_data.pt')
        if not os.path.exists(data_path):
            data = self.style_generation(data_path)
        else:
            data = torch.load(data_path, map_location=self.device)  

        return data
        
    def style_generation(self, data_path):
        def reset_learning_rate(optimizer, new_lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        print(f'Generating {self.cfg.TRAINER.PROMPT_TA.K_PROPMT} style...')
        for i in range(self.cfg.TRAINER.PROMPT_TA.K_PROPMT):
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
                    info += [f"prompt [{i+1}/{self.cfg.TRAINER.PROMPT_TA.K_PROPMT}]"]
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

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        if split == "val" and self.val_loader is not None:
            pass
        else:
            split = "test"
            data_loader = self.test_loader
            print(f"Evaluate on the *test* set")
            for batch_idx, batch in enumerate(data_loader):
                input, label = self.parse_batch_test(batch)
                output = self.model.image_forward(input)  # image feature

                affinity = self.adapter(output)
                cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_value

                class_logits = self.model.classifier(output)
                self.evaluator.process(class_logits + self.alpha * cache_logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return results['accuracy']

    def get_current_lr(self, optim):
        return optim.param_groups[0]["lr"]

    @torch.no_grad()
    def test_dis(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        if split == "val" and self.val_loader is not None:
            pass
        else:
            classifier_weight = weights1 = self.model.classifier.weight.data
            adapter_weight = weights2 = self.adapter.weight.data
            
            # 计算欧氏距离
            euclidean_distance = torch.norm(weights1 - weights2).item()
            print(f'Euclidean Distance: {euclidean_distance}')

            # 计算逐元素绝对差
            absolute_difference = torch.abs(weights1 - weights2)
            mean_absolute_difference = torch.mean(absolute_difference).item()
            print(f'Mean Absolute Difference: {mean_absolute_difference}')

            # 计算逐元素平方差
            squared_difference = (weights1 - weights2) ** 2
            mse = torch.mean(squared_difference).item()
            print(f'Mean Squared Error (MSE): {mse}')

            # 计算 L1 范数
            l1_norm = torch.norm(weights1 - weights2, p=1).item()
            print(f'L1 Norm: {l1_norm}')

            # 计算 L2 范数
            l2_norm = torch.norm(weights1 - weights2, p=2).item()
            print(f'L2 Norm: {l2_norm}')

        return 
    
    @torch.no_grad()
    def test_tsne(self, split=None):        # t-sne visualization
        """A generic t-sne pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        data = self.get_data()
        prompts = data['prompts']
        tokenized_prompts = data['tokenized_prompts']
        labels = data['labels']

        if self.cfg.DATASET.NAME == 'MY_PACS' or self.cfg.DATASET.NAME == 'MY_VLCS' or self.cfg.DATASET.NAME == 'MY_TerraIncognita':
            classes_num = None
        else:
            classes_num = 10
            # classes_num = 65

        if classes_num == None:
            visual_classes = torch.arange(self.n_cls).to(self.device)
            classes_num = self.n_cls
        else:
            visual_classes = torch.arange(classes_num).to(self.device)
            mask = (labels.unsqueeze(1) == visual_classes).any(dim=1)
            prompts = prompts[mask]
            tokenized_prompts = tokenized_prompts[mask]
            labels = labels[mask]
        
        print(f"Getting old features...")
        old_text_features = []
        old_labels = []
        means = torch.zeros((self.n_cls, self.cfg.FEAT_DIM)).to(self.device)
        stds = torch.zeros((self.n_cls, self.cfg.FEAT_DIM)).to(self.device)
        with torch.no_grad():
            for cls in range(self.n_cls):
                text_features = self.model.text_forward(prompts[labels == cls], tokenized_prompts[labels == cls])
                means[cls] = text_features.mean(dim=0)
                stds[cls] = text_features.std(dim=0) 
                old_text_features.append(text_features.float())
                old_labels.append(torch.full((self.cfg.TRAINER.PROMPT_TA.K_PROPMT,), cls))

        old_text_features = torch.cat(old_text_features, dim=0)
        old_labels = torch.cat(old_labels, dim=0).to(self.device)

        print(f"Getting new features...")
        new_text_features = []
        new_labels = []
        for cls in range(self.n_cls):
            epsilon = torch.randn(self.cfg.TRAINER.PROMPT_TA.K_PROPMT, self.cfg.FEAT_DIM).to(self.device)
            new_features = means[cls] + stds[cls] * epsilon
            new_text_features.append(new_features)
            new_labels.append(torch.full((self.cfg.TRAINER.PROMPT_TA.K_PROPMT,), cls+self.n_cls))

        new_text_features = torch.cat(new_text_features, dim=0)
        new_labels = torch.cat(new_labels, dim=0).to(self.device)
        
        # star appendix
        star = True
        if star:
            star_prompts = [name.replace("_", " ") for name in self.dm.dataset.classnames]
            star_tokenized_prompts = clip.tokenize(star_prompts)
            star_text_features = self.model.text_encoder(self.model.prompt_learner.classname_embedding.to(self.device), star_tokenized_prompts.to(self.device))
            star_text_features = star_text_features / star_text_features.norm(dim=-1, keepdim=True)
            star_prompts = self.model.prompt_learner.classname_embedding.to(self.device)
            star_labels = torch.arange(self.n_cls).to(self.device)

            if classes_num != None:
                star_mask = (star_labels.unsqueeze(1) == visual_classes).any(dim=1)
                star_text_features = star_text_features[star_mask]
                star_prompts = star_prompts[star_mask]
                star_labels = star_labels[star_mask]
            
            prompts = torch.cat((prompts, star_prompts), dim=0) 
            text_features = torch.cat((old_text_features, new_text_features, star_text_features), dim=0)
            labels = torch.cat((old_labels, new_labels, star_labels), dim=0)

        tsne = TSNE(
            perplexity=30,
            metric="euclidean",
            n_jobs=8,
            random_state=42,
            verbose=False,
        )
        
        text_features = np.array(text_features.cpu())
        labels = np.array(labels.cpu())

        print('Fitting prompts...')
        embeddings_text = tsne.fit(text_features)
        
        print("Plot figure...")
        dataset_name = self.cfg.DATASET.NAME.split('_')[-1]
        utils.plot_star(embeddings_text, labels, colors=utils.SFDG_COLORS, 
                        name=f'{dataset_name}_text_{len(visual_classes)}', draw_legend=True, 
                        star_length=classes_num, label_name=self.dm.dataset.classnames+self.dm.dataset.classnames)

        return 