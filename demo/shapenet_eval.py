import argparse
import logging
import multiprocessing as mp
import logging
import os
import torch
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
import re

from fvcore.common.file_io import PathManager
from shapenet.config.config import get_shapenet_cfg
from shapenet.utils.coords import project_verts
from meshrcnn.utils.metrics import compare_meshes
from pytorch3d.structures import Meshes
from detectron2.utils.logger import setup_logger

logger = logging.getLogger('demo')

def setup_cfgs(args):
    cfg = get_shapenet_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="MeshRCNN Eval")
    parser.add_argument(
        "--config-file",
        default="configs/shapenet/voxmesh_R50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input", help="A path to main obj folder"
    )
    parser.add_argument(
        "--focal-length", type=float, default=20.0, help="Focal length for the image"
    )
    parser.add_argument(
        "--onlyhighest", action="store_true", help="will return only the highest scoring detection"
    )

    parser.add_argument(
        "--use-combined-mesh", action="store_true", help="use p2m meshrcnn output"
    )

    parser.add_argument(
        "--image-index", type=int, default=1, help="Index of image"
    )

    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def generate_f_v(file, split_xyz=False):
    verts = []
    faces = []
    with open(file, 'r') as f:
        for l in f:
            if l[0] == 'v':
                verts.append(list(map(float, l.split()[1:])))
            elif l[0] == 'f':
                faces.append(list(map(int, l.split()[1:])))
            else:
                continue

    verts = np.array(verts, dtype=np.double)
    faces = np.array(faces, dtype=np.double)
    
    if len(faces) != 0:
        faces -= [1, 1, 1]
    
    if split_xyz:
        x, y, z = np.array_split(verts, 3, axis=1)
        return x, y, z, faces
    
    return torch.from_numpy(verts).float(), torch.from_numpy(faces).float()

def load_gt_data(sd, img_index):
    # read gt mesh
    verts, faces = None, None
    metadata_path = os.path.join(sd, "metadata.pt")
    
    with PathManager.open(metadata_path, "rb") as f:
        metadata = torch.load(f)
    K = metadata["intrinsic"]
    RT = metadata["extrinsics"][img_index]

    mesh_path = os.path.join(sd, "mesh.pt")
    
    with PathManager.open(mesh_path, "rb") as f:
        mesh_data = torch.load(f)

    verts, faces = mesh_data["verts"], mesh_data["faces"]
    verts = project_verts(verts, RT)
    
    verts = verts[None, :, :]
    faces = faces[None, :, :]

    # # print(verts.size(2), faces.size(2))
    gt_mesh = Meshes(verts=verts, faces=faces)

    return gt_mesh

def evaluate(sd, given_img_index):
    to_predict_dir = os.path.join(sd, "final_mesh")
    to_predict_list = sorted(os.listdir(to_predict_dir))
    index = []
    for f in to_predict_list:
        index.append(str(re.search(r"\d+", sd).group()) + '/' + f.split('.')[0])

    df = pd.DataFrame(index=index)
    
    for i, mesh in enumerate(to_predict_list):
        if 'combined' in mesh:
            img_index = given_img_index
        else:
            img_index = int(mesh.split('.')[0][-1])

        gt_mesh = load_gt_data(sd, img_index)
        mesh_dir = os.path.join(to_predict_dir, mesh)
        pred_verts, pred_faces = generate_f_v(mesh_dir)

        pred_verts = pred_verts[None, :, :]
        pred_faces = pred_faces[None, :, :]
        pred_mesh = Meshes(verts=pred_verts, faces=pred_faces)

        cur_metrics = compare_meshes(
            pred_mesh, gt_mesh, 
            #  scale=0.57, thresholds=[0.01, 0.014142], 
            reduce=False)

        cur_metrics["verts_per_mesh"] = pred_mesh.num_verts_per_mesh().cpu()
        cur_metrics["faces_per_mesh"] = pred_mesh.num_faces_per_mesh().cpu()

        for k, v in cur_metrics.items():
            df.at[index[i], k] = v.item()

    save_dir = os.path.join(sd, "eval.csv")
    df.to_csv(save_dir)
    logger.info(" - Predictions saved for %s" %(save_dir))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    device = torch.device("cuda:%d" % 0)
    logger = setup_logger(name="shapenet eval")
    logger.info("Arguments: " + str(args))

    main_dir = os.path.join(args.input)
    cat_folders = sorted(os.listdir(main_dir))

    for cd in cat_folders[:]:
        sub_dir = os.path.join(main_dir, cd)
        sub_folders = sorted(os.listdir(sub_dir), key=lambda x: int(x))
        logger.info("In %s" %(sub_dir))
        
        for d in sub_folders[:]:
            sd = os.path.join(sub_dir, d)
            evaluate(sd, given_img_index=args.image_index)

    

    # meshrcnn_output_dir = os.path.join(args.input_eval_folder, "final_mesh", "bench0.obj")
    # p2m_mrcnn_output_dir = os.path.join(args.input_eval_folder, "final_mesh", "bench_combined1.obj")

    # if args.use_combined_mesh:
    #     f = p2m_mrcnn_output_dir
    # else:
    #     f = meshrcnn_output_dir

    # pred_verts, pred_faces = generate_f_v(f)
    
    # print(pred_verts.shape)
    # print(pred_faces.shape)

    # pred_verts = pred_verts[None, :, :]
    # pred_faces = pred_faces[None, :, :]
    # pred_mesh = Meshes(verts=pred_verts, faces=pred_faces)

    # cur_metrics = compare_meshes(pred_mesh, gt_mesh, 
    #                             #  scale=0.57, thresholds=[0.01, 0.014142], 
    #                              reduce=False)
    # cur_metrics["verts_per_mesh"] = pred_mesh.num_verts_per_mesh().cpu()
    # cur_metrics["faces_per_mesh"] = pred_mesh.num_faces_per_mesh().cpu()

    # logger.info('Eval for %s' % (f.split('/')[-1]))
    # for k, v in cur_metrics.items():
    #     print(f'{k:<25}: {v.item()}')