import argparse
import logging
import multiprocessing as mp
import logging
import os
import torch
import torch.multiprocessing as mp
from fvcore.common.file_io import PathManager

from shapenet.config.config import get_shapenet_cfg
from shapenet.utils.coords import project_verts
from meshrcnn.utils.metrics import compare_meshes

import numpy as np
from pytorch3d.structures import Meshes

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
    parser.add_argument("--input-gt", help="A path to ground truth mesh")
    parser.add_argument("--input-pred", help="A path to generated mesh")
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

def generate_f_v(file, split_xyz=False):
    verts = []
    faces = []
    with open(file, 'r') as f:
        for l in f:
            if l[0] == 'v':
                verts.append(list(map(float, l.split()[1:])))
            else:
                faces.append(list(map(int, l.split()[1:])))
    
    verts = np.array(verts, dtype=np.double)
    faces = np.array(faces, dtype=np.double)
    
    if len(faces) != 0:
        faces -= [1, 1, 1]
    
    if split_xyz:
        x, y, z = np.array_split(verts, 3, axis=1)
        return x, y, z, faces
    
    return torch.from_numpy(verts).float(), torch.from_numpy(faces).float()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    device = torch.device("cuda:%d" % 0)

    # read gt mesh
    verts, faces = None, None
    temp_path = '/home/hengkaiz/Documents/fyp/meshrcnn/datasets/shapenet/ShapeNetV1processed/04256520/1a4a8592046253ab5ff61a3a2a0e2484'

    metadata_path = os.path.join(temp_path, "metadata.pt")
    
    with PathManager.open(metadata_path, "rb") as f:
        metadata = torch.load(f)
    K = metadata["intrinsic"]
    RT = metadata["extrinsics"][0]

    mesh_path = os.path.join(temp_path, "mesh.pt")
    
    with PathManager.open(mesh_path, "rb") as f:
        mesh_data = torch.load(f)

    verts, faces = mesh_data["verts"], mesh_data["faces"]
    verts = project_verts(verts, RT)
    
    verts = verts[None, :, :]
    faces = faces[None, :, :]

    # # print(verts.size(2), faces.size(2))
    gt_mesh = Meshes(verts=verts, faces=faces)

    pred_verts, pred_faces = generate_f_v('output_demo/sofa1.obj')
    
    pred_verts = pred_verts[None, :, :]
    pred_faces = pred_faces[None, :, :]
    pred_mesh = Meshes(verts=pred_verts, faces=pred_faces)

    cur_metrics = compare_meshes(pred_mesh, gt_mesh, reduce=False)
    cur_metrics["verts_per_mesh"] = pred_mesh.num_verts_per_mesh().cpu()
    cur_metrics["faces_per_mesh"] = pred_mesh.num_faces_per_mesh().cpu()

    for k, v in cur_metrics.items():
        print(f'{k:<25}: {v.item()}')