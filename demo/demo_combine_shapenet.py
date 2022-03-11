import argparse
import logging
import multiprocessing as mp
import logging
import os
from detectron2.evaluation import inference_context
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger
from fvcore.common.file_io import PathManager

from pytorch3d.io import save_obj

from shapenet.config.config import get_shapenet_cfg
from shapenet.data.utils import imagenet_preprocess
from shapenet.modeling.heads import voxel_head
from shapenet.modeling.mesh_arch import build_model
from shapenet.utils.checkpoint import clean_state_dict

import torchvision.transforms as T
from PIL import Image

import glob

logger = logging.getLogger('demo')

def setup_cfgs(args):
    cfg = get_shapenet_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="MeshRCNN Demo")
    parser.add_argument(
        "--config-file",
        default="configs/shapenet/voxmesh_R50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", help="Path to main obj folder")
    parser.add_argument("--new-projection", type=int, default=1, help="Which img to use for new projection")
    parser.add_argument(
        "--focal-length", type=float, default=20.0, help="Focal length for the image"
    )
    parser.add_argument(
        "--onlyhighest", action="store_true", help="will return only the highest scoring detection"
    )

    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def load_img(sd, po):
    device = torch.device("cuda:%d" % 0)

    img_list = sorted(glob.glob(os.path.join(sd, "*.png")))
    first_img = os.path.join(img_list[po])

    # load image
    transform = [T.ToTensor()]
    transform.append(imagenet_preprocess())
    transform = T.Compose(transform)
    
    im_name = first_img.split("/")[-1].split(".")[0]

    with PathManager.open(first_img, "rb") as f:
        img = Image.open(f).convert("RGB")
    img = transform(img)
    img = img[None, :, :, :]
    img = img.to(device)

    return img

def load_intermediate(sd):
    int_dir = os.path.join(sd, "intermediate_mesh")
    int_list = sorted(os.listdir(int_dir))
    int_mesh = torch.load(os.path.join(int_dir, int_list[0]))

    return int_mesh

def load_img_feats(sd, po):
    img_dir = os.path.join(sd, "img_feat")
    img_feat_list = sorted(os.listdir(img_dir))
    img_feat = torch.load(os.path.join(img_dir, img_feat_list[po]))

    return img_feat

def load_p(sd, po):
    P_dir = os.path.join(sd, "P")
    P_list = sorted(os.listdir(P_dir))
    P = torch.load(os.path.join(P_dir, P_list[po]))

    return P

def refine_mesh(sd, po=1):
    img = load_img(sd, po)
    img_feat = load_img_feats(sd, po)
    int_mesh = load_intermediate(sd)
    P = load_p(sd, po)

    with inference_context(model):
        meshes_pred = model(img, combine=True, cimg_feats=img_feat, fmesh=int_mesh, cP=P, vox=None)

    verts, faces = meshes_pred[-1].get_mesh_verts_faces(0)

    save_file = os.path.join(sd, "final_mesh", f"bench_combined{po}.obj")
    save_obj(save_file, verts, faces)
    logger.info("- Predictions saved in %s" % (save_file))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    device = torch.device("cuda:%d" % 0)

    logger = setup_logger(name="demo shapenet")
    logger.info("Arguments: " + str(args))

    cfg = setup_cfgs(args)

    # load checkpoing and build model
    if cfg.MODEL.CHECKPOINT == "":
        raise ValueError("Invalid checkpoint provided")

    logger.info("Loading model from checkpoint: %s" % (cfg.MODEL.CHECKPOINT))
    cp = torch.load(PathManager.get_local_path(cfg.MODEL.CHECKPOINT))
    state_dict = clean_state_dict(cp["best_states"]["model"])
    model = build_model(cfg)
    model.load_state_dict(state_dict)
    logger.info("Model loaded")
    model.to(device)

    main_dir = os.path.join(args.input)
    cat_folders = sorted(os.listdir(main_dir))

    for cd in cat_folders:
        sub_dir = os.path.join(main_dir, cd)
        sub_folders = sorted(os.listdir(sub_dir), key=lambda x: int(x))
        logger.info("In %s" %(sub_dir))
        
        for d in sub_folders[:]:
            sd = os.path.join(sub_dir, d)

            refine_mesh(sd, po=args.new_projection)