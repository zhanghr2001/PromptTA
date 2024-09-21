import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
from datasets import *
from trainers import *


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    
    cfg.MODEL_DIR = ""
    if args.model_dir:
        cfg.MODEL_DIR = args.model_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    
    if cfg.MODEL.BACKBONE.NAME == "RN50":   # embedding dimension size for image feature
        cfg.FEAT_DIM = 1024
    elif cfg.MODEL.BACKBONE.NAME == "ViT-B/16":
        cfg.FEAT_DIM = 512
    elif cfg.MODEL.BACKBONE.NAME == "ViT-L/14":
        cfg.FEAT_DIM = 768

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.gpu:
        cfg.GPU = args.gpu
        
    if args.target_domains:
        cfg.SAVE_MODEL = args.save
        if "PACS" in cfg.DATASET.NAME:
            DOMAINS = {'a': "art_painting", 'c':"cartoon", 'p':"photo", 's':"sketch"}
            cfg.ALPHA = 1
            cfg.BETA = 2
        elif "VLCS" in cfg.DATASET.NAME:
            DOMAINS = {'c': "caltech", 'l':"labelme", 'p':"pascal", 's':"sun"}
            cfg.ALPHA = 2
            cfg.BETA = 2
        elif "OfficeHome" in cfg.DATASET.NAME:
            DOMAINS = {'a': "art", 'c':"clipart", 'p':"product", 'r':"real_world"}
            cfg.ALPHA = 1
            cfg.BETA = 2
        elif "DomainNet" in cfg.DATASET.NAME:
            DOMAINS = {'c': "clipart", 'i':"infograph", 'p':"painting", 'q':"quickdraw", 'r':"real", 's':"sketch"}
            cfg.ALPHA = 2
            cfg.BETA = 2
        else:
            raise ValueError
        
        cfg.ALL_DOMAINS = list(DOMAINS.keys())
        cfg.TARGET_DOMAIN = args.target_domains[0]
        cfg.DATASET.TARGET_DOMAINS = [DOMAINS[cfg.TARGET_DOMAIN]]
        DOMAINS.pop(cfg.TARGET_DOMAIN)
        cfg.SOURCE_DOMAINS = list(DOMAINS.keys())
        cfg.DATASET.SOURCE_DOMAINS = list(DOMAINS.values())


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    
    cfg.MODEL.BACKBONE.PATH = './assets'       # path of pretrained CLIP model
    cfg.TEST.FINAL_MODEL = "last"           # best_val or last
    cfg.TEST.DO_TEST = True

    cfg.OPTIM_PROMPT = cfg.OPTIM.clone()
    cfg.OPTIM_CLASSIFIER = cfg.OPTIM.clone()
    cfg.OPTIM_ADAPTER = cfg.OPTIM.clone()

    if 'CLIP' in args.trainer:
        cfg.TRAINER.CLIP = CN()
        cfg.TRAINER.CLIP.PREC = "fp16"  # fp16, fp32
    
    elif 'PROMPT_STYLER' in args.trainer:    
        cfg.TRAINER.PROMPT_STYLER = CN()
        cfg.TRAINER.PROMPT_STYLER.PREC = "fp16"       # fp16, fp32  
        cfg.TRAINER.PROMPT_STYLER.N_CTX = 1          # number of text context vectors
        cfg.TRAINER.PROMPT_STYLER.CSC = False         # class-specific context
        cfg.TRAINER.PROMPT_STYLER.CTX_INIT = 'A S style of a'
        cfg.TRAINER.PROMPT_STYLER.K_PROPMT = 80
        cfg.TRAINER.PROMPT_STYLER.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
        
    elif 'PROMPT_TA' in args.trainer:
        cfg.TRAINER.PROMPT_TA = CN()
        cfg.TRAINER.PROMPT_TA.PREC = "fp16"       # fp16, fp32  
        cfg.TRAINER.PROMPT_TA.N_CTX = 1           # number of text context vectors
        cfg.TRAINER.PROMPT_TA.CSC = False         # class-specific context
        cfg.TRAINER.PROMPT_TA.K_PROPMT = 80
        cfg.TRAINER.PROMPT_TA.CTX_INIT = 'A S style of a'   # initialization words
        cfg.TRAINER.PROMPT_TA.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    else:
        raise NotImplementedError
        

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.epoch)
        trainer.test()
        return
    
    if args.tSNE:
        trainer.load_model(args.model_dir, epoch=args.epoch)
        trainer.test_tsne()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/opt/data/private/OOD_data", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="output directory")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="", help="path to config file for dataset setup")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domain for DG")
    
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    
    parser.add_argument("--trainer", type=str, default="PROMPT_TA", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")

    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument("--epoch", type=int, default=50, help="load model")
    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")
    parser.add_argument("--save", type=str, default=False, help="need to save model")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")

    parser.add_argument("--tSNE", action="store_true", help="t-SNE visualization")

    args = parser.parse_args()
    
    main(args)
