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

import trimesh
import pyvista as pv
import pyacvd

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
    parser.add_argument("--input", help="A path to an input image")
    parser.add_argument("--output", help="A directory to save output visualizations")
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

def resample_mesh(mesh, count=2466):
    pv_mesh = pv.wrap(mesh)
    logger.info('Original mesh:')
    print(pv_mesh)
    
    clus = pyacvd.Clustering(pv_mesh)
    clus.subdivide(3)
    clus.cluster(count)

    # remesh
    remesh = clus.create_mesh()
    logger.info('Resampled mesh:\n')
    print(remesh)

    verts = remesh.points
    faces = remesh.faces.reshape((-1, 4))[:, 1:]
    
    return verts, faces

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    device = torch.device("cuda:%d" % 0)

    logger = setup_logger(name="demo shapenet")
    logger.info("Arguments: " + str(args))

    cfg = setup_cfgs(args)

    # load checkpoing and build model
    if cfg.MODEL.CHECKPOINT == "":
        raise ValueError("Invalid checkpoing provided")
    logger.info("Loading model from checkpoint: %s" % (cfg.MODEL.CHECKPOINT))
    cp = torch.load(PathManager.get_local_path(cfg.MODEL.CHECKPOINT))
    state_dict = clean_state_dict(cp["best_states"]["model"])
    model = build_model(cfg)
    model.load_state_dict(state_dict)
    logger.info("Model loaded")
    model.to(device)

    # load image
    transform = [T.ToTensor()]
    transform.append(imagenet_preprocess())
    transform = T.Compose(transform)
    
    im_name = args.input.split("/")[-1].split(".")[0]

    with PathManager.open(args.input, "rb") as f:
        img = Image.open(f).convert("RGB")
    img = transform(img)
    img = img[None, :, :, :]
    img = img.to(device)

    with inference_context(model):
        img_feats, _, meshes_pred = model(img)

    verts, faces = meshes_pred[-1].get_mesh_verts_faces(0)

    mesh = trimesh.Trimesh(vertices=verts.cpu().detach().numpy(), faces=faces.cpu().detach().numpy())
    resampled_verts, resampled_faces = resample_mesh(mesh)


    save_file = os.path.join(args.output, "%s.obj" % (im_name))
    save_obj(save_file, torch.from_numpy(resampled_verts), torch.from_numpy(resampled_faces))
    logger.info("Predictions saved in %s" % (save_file))
