import argparse
from audioop import reverse
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import tensorflow as tf
import os
import pyvista as pv
import pyacvd
import math
import glob

import torch
import torch.multiprocessing as mp

import logging

from pytorch3d.structures import Meshes
from pathlib import Path

from detectron2.utils.logger import setup_logger

tf.compat.v1.disable_eager_execution()

def get_parser():
    parser = argparse.ArgumentParser(description="MeshRCNN Demo")
    parser.add_argument(
        "--config-file",
        default="configs/shapenet/voxmesh_R50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input", help="Path to input main folder"
    )
    parser.add_argument(
        "--focal-length", type=float, default=20.0, help="Focal length for the image"
    )

    parser.add_argument(
        "--avg-points", action="store_true", help="Avg distance between points"
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
    
    verts = np.array(verts)
    faces = np.array(faces)
    
    if len(faces) != 0:
        faces -= [1, 1, 1]
    
    if split_xyz:
        x, y, z = np.array_split(verts, 3, axis=1)
        return x, y, z, faces
    
    return verts, faces

def combine_mesh(mesh1, mesh2):
    mesh_list = [mesh1, mesh2]

    # hard merge
    vertice_list = [mesh.vertices for mesh in mesh_list]
    faces_list = [mesh.faces for mesh in mesh_list]
    faces_offset = np.cumsum([v.shape[0] for v in vertice_list])
    faces_offset = np.insert(faces_offset, 0, 0)[:-1]

    vertices = np.vstack(vertice_list)
    faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)])

    return vertices

# cameras
# azimuth, elevation, in-plane rotation, distance, field of view.
def normal(v):
    norm = tf.norm(v)
    if norm == 0:
        return v
    return tf.divide(v, norm)

def cameraMat(param):
    theta = param[0] * np.pi / 180.0
    camy = param[3] * tf.sin(param[1] * np.pi / 180.0)
    lens = param[3] * tf.cos(param[1] * np.pi / 180.0)
    camx = lens * tf.cos(theta)
    camz = lens * tf.sin(theta)
    Z = tf.stack([camx, camy, camz])

    x = camy * tf.cos(theta + np.pi)
    z = camy * tf.sin(theta + np.pi)
    Y = tf.stack([x, lens, z])
    X = tf.linalg.cross(Y, Z)

    cm_mat = tf.stack([normal(X), normal(Y), normal(Z)])
    return cm_mat, Z

def camera_trans_inv(camera_metadata, xyz):
    c, o = cameraMat(camera_metadata)
    inv_xyz = (tf.matmul(xyz, tf.linalg.inv(tf.transpose(c)))) + o
    return inv_xyz

def camera_trans(camera_metadata, xyz):
    c, o = cameraMat(camera_metadata)
    points = xyz[:, :]
    pt_trans = points - o
    pt_trans = tf.matmul(pt_trans, tf.transpose(c))
    return pt_trans

def gen_projection(origin_cam, new_cam, origin_verts):
    point_origin = camera_trans_inv(origin_cam, origin_verts)
    new_verts = camera_trans(new_cam, point_origin)
    
    return new_verts

def resample(mesh, count=2118):
    pv_mesh = pv.wrap(mesh)
    
    clus = pyacvd.Clustering(pv_mesh)
    clus.subdivide(1)
    clus.cluster(count)

    # remesh
    remesh = clus.create_mesh()
    
    return remesh

def gen_projected_mesh(sub_dir):
    objs = sorted(glob.glob(sub_dir + '/cube_mesh/*.obj'))
    camera = np.loadtxt(os.path.join(sub_dir, 'cameras.txt'))[:2]

    vert1, face1 = generate_f_v(objs[0])
    vert2, face2 = generate_f_v(objs[1])

    res_projected = gen_projection(origin_cam=camera[0], new_cam=camera[1], origin_verts=vert1)

    with tf.compat.v1.Session() as sess:
        res_projected_moved = res_projected.eval()

    OLD = res_projected_moved
    NEW = vert2
    SAMPLES = 10

    # scaling
    v_min = np.array([
        np.median(NEW[:, 0][np.argsort(NEW[:, 0])[:SAMPLES]]), 
        np.median(NEW[:, 1][np.argsort(NEW[:, 1])[:SAMPLES]]), 
        np.median(NEW[:, 2][np.argsort(NEW[:, 2])[:SAMPLES]])])

    v_max = np.array([
        np.median(np.median(NEW[:, 0][np.argsort(NEW[:, 0])[-SAMPLES:]])),
        np.median(np.median(NEW[:, 1][np.argsort(NEW[:, 1])[-SAMPLES:]])), 
        np.median(np.median(NEW[:, 2][np.argsort(NEW[:, 2])[-SAMPLES:]]))])

    v = OLD

    v_old_min = np.array([
        np.median(OLD[:, 0][np.argsort(OLD[:, 0])[:SAMPLES]]), 
        np.median(OLD[:, 1][np.argsort(OLD[:, 1])[:SAMPLES]]), 
        np.median(OLD[:, 2][np.argsort(OLD[:, 2])[:SAMPLES]])])

    v_old_max = np.array([
        np.median(np.median(OLD[:, 0][np.argsort(OLD[:, 0])[-SAMPLES:]])),
        np.median(np.median(OLD[:, 1][np.argsort(OLD[:, 1])[-SAMPLES:]])), 
        np.median(np.median(OLD[:, 2][np.argsort(OLD[:, 2])[-SAMPLES:]]))])

    test_scaled = ((v - v_old_min) * (v_max - v_min) / (v_old_max - v_old_min)) + v_min

    mesh1 = trimesh.Trimesh(vertices=test_scaled, faces=face1)
    mesh2 = trimesh.Trimesh(vertices=vert2, faces=face2)

    pv_mesh1 = pv.wrap(mesh1)
    pv_mesh2 = pv.wrap(mesh2)

    resample1 = resample(pv_mesh1, pv_mesh2.n_points)
    test_scaled = resample1.points
    
    seen = set()
    match = {}

    for x1 in test_scaled:
        min_dist = float('inf')
        min_vert = []
        
        for x2 in vert2:
            if tuple(x2) in seen:
                continue
            
            dist = np.linalg.norm(x1-x2)
            if dist < min_dist:
                min_dist = dist
                min_vert = x2
                
        match[tuple(min_vert)] = x1
        seen.add(tuple(x2))

    new_verts = []

    for x in vert2:
        if match.get(tuple(x), None) is not None:
            new_verts.append((match.get(tuple(x)) + x) / 2)
        else:
            new_verts.append(x)

    new_verts = np.array(new_verts)

    final_mesh = trimesh.Trimesh(vertices=new_verts, faces=face2)
    pv_final = pv.wrap(final_mesh)

    verts = pv_final.points
    faces = pv_final.faces.reshape((-1, 4))[:, 1:]

    verts = np.array(verts)
    faces = np.array(faces)

    verts = np.expand_dims(verts, axis=0)
    faces = np.expand_dims(faces, axis=0)

    verts = torch.from_numpy(verts).to('cuda', dtype=torch.float32)
    faces = torch.from_numpy(faces).to('cuda', dtype=torch.int64)

    mesh = Meshes(verts=verts, faces=faces)
    save_dir = os.path.join(sub_dir, 'intermediate_mesh')

    if not Path(save_dir).is_dir():
        os.mkdir(save_dir)

    torch.save(mesh.cuda(), 
        os.path.join(save_dir, 'intermediate_mesh.pt'))

    logger.info(" - Predictions saved for %s" %(sub_dir))

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    device = torch.device("cuda:%d" % 0)
    logger = setup_logger(name="intermediate projection")
    logger.info("Arguments: " + str(args))

    main_dir = os.path.join(args.input)
    cat_folders = sorted(os.listdir(main_dir))

    for cd in cat_folders:
        sub_dir = os.path.join(main_dir, cd)
        sub_folders = sorted(os.listdir(sub_dir), key=lambda x: int(x))
        logger.info("In %s" %(sub_dir))
        
        for d in sub_folders[:]:
            sd = os.path.join(sub_dir, d)

            gen_projected_mesh(sd)