from torch.cuda.amp import GradScaler

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip

from trainers.base_sfdg import *
from utils.clip_part import *
from utils.templates import CUSTOM_TEMPLATES


class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.text_encoder = Simple_TextEncoder(clip_model)

        self.image_encoder = clip_model.visual
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        prompts = [c.replace("_", " ") for c in classnames]
        self.tokenized_prompts = clip.tokenize(prompts)
    
    def forward(self, image):
        text_features = self.text_encoder(self.tokenized_prompts.to(self.logit_scale.device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t().cuda(image_features.device)

        return logits


@TRAINER_REGISTRY.register()
class CLIP_ZS_C(Base_SFDG):
    """
    ZS: Zero-Shot CLIP
    """  
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.save = cfg.SAVE_MODEL

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CLIP.PREC == "fp32" or cfg.TRAINER.CLIP.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print("# params: {:,}".format(0))

        self.model.to(self.device)

        # no loss
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CLIP_model", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.CLIP.PREC == "amp" else None
    
    def train(self):
        self.before_train()
        self.after_train()
    
        