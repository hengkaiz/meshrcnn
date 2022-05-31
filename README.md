# Mesh R-CNN++

## Installation Requirements
Mesh R-CNN++ is build onto of Mesh R-CNN,

The following packages are needed:
- [Detectron2][d2]
- [PyTorch3D][py3d]

The implementation of Mesh R-CNN is based on [Detectron2][d2] and [PyTorch3D][py3d].
You will first need to install those in order to be able to run Mesh R-CNN.

To install
```
git clone https://github.com/facebookresearch/meshrcnn.git
cd meshrcnn && pip install -e .
```

## Demo

### Generate Dataset

Download ShapeNet Dataset
See [INSTRUCTIONS_SHAPENET.md](INSTRUCTIONS_SHAPENET.md) for more instructions.

Process ShapeNet Dataset for Mesh R-CNN++
```
python tools/gen_dataset.py
```

### Run Mesh R-CNN++ with 2 input images

Run Mesh Generation Stage

```
python demo/demo_shapenet.py --input final_eval_data \
--config-file configs/shapenet/voxmesh_R50.yaml \
MODEL.CHECKPOINT shapenet://voxmesh_R50.pth
```
Run Mesh Projection Stage

```
python tools/gen_intermediate_projection.py --input final_eval_data
```

Run Mesh Refinement Stage
```
python demo/demo_combine_shapenet.py --input final_eval_data/ \
--config-file configs/shapenet/voxmesh_R50.yaml \
MODEL.CHECKPOINT shapenet://voxmesh_R50.pth
```

Run evaluation
```
python demo/shapenet_eval.py --input final_eval_data/ --image-index 1
```
