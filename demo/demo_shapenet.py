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
from pathlib import Path
from pytorch3d.io import save_obj

from shapenet.config.config import get_shapenet_cfg
from shapenet.data.utils import imagenet_preprocess
from shapenet.modeling.heads import voxel_head
from shapenet.modeling.mesh_arch import build_model
from shapenet.utils.checkpoint import clean_state_dict

import torchvision.transforms as T

import glob
from PIL import Image

import trimesh
import pyvista as pv
import pyacvd
import numpy as np

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
    parser.add_argument("--input", help="A path to an input main folder")
    # parser.add_argument("--output", help="A directory to save output visualizations")
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
    # logger.info('Original mesh:')
    # print(pv_mesh)
    
    clus = pyacvd.Clustering(pv_mesh)
    clus.subdivide(3)
    clus.cluster(count)

    # remesh
    remesh = clus.create_mesh()

    # verts = remesh.points
    # faces = remesh.faces.reshape((-1, 4))[:, 1:]
    
    return remesh

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

    sub_dir = sorted(os.listdir(args.input))

    for sd in sub_dir:
        curr_path = os.path.join(args.input, sd)
        images = glob.glob(curr_path + "/*.png")
        
        for img_dir in images:
            # load image
            transform = [T.ToTensor()]
            transform.append(imagenet_preprocess())
            transform = T.Compose(transform)
            
            im_name = img_dir.split("/")[-1].split(".")[0]

            with PathManager.open(img_dir, "rb") as f:
                img = Image.open(f).convert("RGB")

            img = transform(img)
            img = img[None, :, :, :]
            img = img.to(device)

            with inference_context(model):
                img_feats, voxel_scores, meshes_pred, P, cubified_meshes = model(img)

            # Save voxel_score
            voxel_odir = os.path.join(curr_path, "voxel_score")
            if not Path(voxel_odir).is_dir():
                os.mkdir(voxel_odir)

            voxel_file = os.path.join(voxel_odir, "%s.pt" % (im_name))
            torch.save(voxel_scores, voxel_file)

            # Save image features
            imgfeat_odir = os.path.join(curr_path, "img_feat")
            if not Path(imgfeat_odir).is_dir():
                os.mkdir(imgfeat_odir)

            img_feat_file = os.path.join(imgfeat_odir, "%s.pt" % (im_name))
            torch.save(img_feats, img_feat_file)

            # Save P
            p_odir = os.path.join(curr_path, "P")
            if not Path(p_odir).is_dir():
                os.mkdir(p_odir)

            p_file = os.path.join(p_odir, "%s.pt" % (im_name))
            torch.save(P, p_file)

            # Save cubified mesh
            cmesh_odir = os.path.join(curr_path, "cube_mesh")
            if not Path(cmesh_odir).is_dir():
                os.mkdir(cmesh_odir)

            cube_mesh_file = os.path.join(cmesh_odir, "%s_cube.obj" % (im_name))
            c_verts, c_faces = cubified_meshes[-1].get_mesh_verts_faces(0)
            save_obj(cube_mesh_file, c_verts, c_faces)

            # Save predicted mesh
            mesh_odir = os.path.join(curr_path, "final_mesh")
            if not Path(mesh_odir).is_dir():
                os.mkdir(mesh_odir)

            save_file = os.path.join(mesh_odir, "%s.obj" % (im_name))
            verts, faces = meshes_pred[-1].get_mesh_verts_faces(0)
            save_obj(save_file, verts, faces)
            logger.info("Predictions saved for %s/%s" % (curr_path.split('/')[-1], im_name))
