# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from detectron2.utils.registry import Registry
from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from shapenet.modeling.backbone import build_backbone
from shapenet.modeling.heads import MeshRefinementHead, VoxelHead
from shapenet.utils.coords import get_blender_intrinsic_matrix, voxel_to_world

import tensorflow as tf
import numpy as np

MESH_ARCH_REGISTRY = Registry("MESH_ARCH")


@MESH_ARCH_REGISTRY.register()
class VoxMeshHead(nn.Module):
    def __init__(self, cfg):
        super(VoxMeshHead, self).__init__()

        # fmt: off
        backbone                = cfg.MODEL.BACKBONE
        self.cubify_threshold   = cfg.MODEL.VOXEL_HEAD.CUBIFY_THRESH
        self.voxel_size         = cfg.MODEL.VOXEL_HEAD.VOXEL_SIZE
        # fmt: on

        self.register_buffer("K", get_blender_intrinsic_matrix())
        # backbone
        self.backbone, feat_dims = build_backbone(backbone)
        # voxel head
        cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS = feat_dims[-1]
        self.voxel_head = VoxelHead(cfg)
        # mesh head
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = sum(feat_dims)
        self.mesh_head = MeshRefinementHead(cfg)

    def _get_projection_matrix(self, N, device):
        return self.K[None].repeat(N, 1, 1).to(device).detach()

    def _dummy_mesh(self, N, device):
        verts_batch = torch.randn(N, 4, 3, device=device)
        faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
        faces = torch.tensor(faces, dtype=torch.int64)
        faces_batch = faces.view(1, 4, 3).expand(N, 4, 3).to(device)
        return Meshes(verts=verts_batch, faces=faces_batch)

    def cubify(self, voxel_scores):
        V = self.voxel_size
        N = voxel_scores.shape[0]
        voxel_probs = voxel_scores.sigmoid()
        active_voxels = voxel_probs > self.cubify_threshold
        voxels_per_mesh = (active_voxels.view(N, -1).sum(dim=1)).tolist()
        start = V // 4
        stop = start + V // 2
        for i in range(N):
            if voxels_per_mesh[i] == 0:
                voxel_probs[i, start:stop, start:stop, start:stop] = 1
        meshes = cubify(voxel_probs, self.cubify_threshold)

        meshes = self._add_dummies(meshes)
        meshes = voxel_to_world(meshes)
        return meshes

    def _add_dummies(self, meshes):
        N = len(meshes)
        dummies = self._dummy_mesh(N, meshes.device)
        verts_list = meshes.verts_list()
        faces_list = meshes.faces_list()
        for i in range(N):
            if faces_list[i].shape[0] == 0:
                vv, ff = dummies.get_mesh(i)
                verts_list[i] = vv
                faces_list[i] = ff
        return Meshes(verts=verts_list, faces=faces_list)

    def forward(self, imgs, voxel_only=False, combine=False, camera=None, cimg_feats=None, fmesh=None, cP=None, vox=None):
        if combine:
            # cubified_meshes = self.cubify(vox)
            refined_meshes = self.mesh_head(cimg_feats, fmesh, cP)

            return refined_meshes
            # return cubified_meshes
        
        N = imgs.shape[0]
        device = imgs.device

        img_feats = self.backbone(imgs)
        voxel_scores = self.voxel_head(img_feats[-1])
        P = self._get_projection_matrix(N, device)
        
        if voxel_only:
            dummy_meshes = self._dummy_mesh(N, device)
            dummy_refined = self.mesh_head(img_feats, dummy_meshes, P)

            print('yessir:', len(voxel_scores), len(dummy_refined))
            return voxel_scores, dummy_refined

        cubified_meshes = self.cubify(voxel_scores)
        refined_meshes = self.mesh_head(img_feats, cubified_meshes, P)
        return img_feats, voxel_scores, refined_meshes, P, cubified_meshes

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

@MESH_ARCH_REGISTRY.register()
class SphereInitHead(nn.Module):
    def __init__(self, cfg):
        super(SphereInitHead, self).__init__()

        # fmt: off
        backbone                = cfg.MODEL.BACKBONE
        self.ico_sphere_level   = cfg.MODEL.MESH_HEAD.ICO_SPHERE_LEVEL
        # fmt: on

        self.register_buffer("K", get_blender_intrinsic_matrix())
        # backbone
        self.backbone, feat_dims = build_backbone(backbone)
        # mesh head
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = sum(feat_dims)
        self.mesh_head = MeshRefinementHead(cfg)

    def _get_projection_matrix(self, N, device):
        return self.K[None].repeat(N, 1, 1).to(device).detach()

    def forward(self, imgs):
        N = imgs.shape[0]
        device = imgs.device

        img_feats = self.backbone(imgs)
        P = self._get_projection_matrix(N, device)

        init_meshes = ico_sphere(self.ico_sphere_level, device).extend(N)
        refined_meshes = self.mesh_head(img_feats, init_meshes, P)
        return None, refined_meshes


@MESH_ARCH_REGISTRY.register()
class Pixel2MeshHead(nn.Module):
    def __init__(self, cfg):
        super(Pixel2MeshHead, self).__init__()

        # fmt: off
        backbone                = cfg.MODEL.BACKBONE
        self.ico_sphere_level   = cfg.MODEL.MESH_HEAD.ICO_SPHERE_LEVEL
        # fmt: on

        self.register_buffer("K", get_blender_intrinsic_matrix())
        # backbone
        self.backbone, feat_dims = build_backbone(backbone)
        # mesh head
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = sum(feat_dims)
        self.mesh_head = MeshRefinementHead(cfg)

    def _get_projection_matrix(self, N, device):
        return self.K[None].repeat(N, 1, 1).to(device).detach()

    def forward(self, imgs):
        N = imgs.shape[0]
        device = imgs.device

        img_feats = self.backbone(imgs)
        P = self._get_projection_matrix(N, device)

        init_meshes = ico_sphere(self.ico_sphere_level, device).extend(N)
        refined_meshes = self.mesh_head(img_feats, init_meshes, P, subdivide=True)
        return None, refined_meshes


def build_model(cfg):
    name = cfg.MODEL.MESH_HEAD.NAME
    return MESH_ARCH_REGISTRY.get(name)(cfg)
