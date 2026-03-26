# Environment Setup

**1. Download nuScenes**

Download the [nuScenes dataset](https://www.nuscenes.org/download) to `./data/nuscenes`.

**2. Download infos files**

Download the Info Files from our [huggingface datasets](https://huggingface.co/datasets/pqh22/ColaVLA).

Here, we annotate each trajectory sample with meta-action labels based on vehicle driving signals and GT trajectories. The mapping between label values and specific driving strategies can be found in the **MAPPING_V3** vocabulary in [transform_3d.py](../projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py).

**3. Pretrained Weights**
```shell
cd /path/to/ColaVLA
mkdir ckpts
```
The related checkpoints and data files can be downloaded from our [hugging face dataset page](https://huggingface.co/datasets/pqh22/ColaVLA).


**4. Install Packages**
```shell
cd /path/to/ColaVLA
conda create -n colavla python=3.9
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.3.2
pip install transformers==4.31.0 
pip install mmcv-full==1.7.0
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6 
pip install -e .
cd ..
git clone https://github.com/OpenDriveLab/OpenLane-V2.git
cd OpenLane-V2
pip install -e .
cd ..
pip install -r requirements.txt
```
We also provide detailed package versions in [environment.yml](./environment.yml).

After preparation, you will be able to see the following directory structure:  

**5. Folder structure**
```
ColaVLA
├── projects/
├── mmdetection3d/
├── OpenLane-V2/
├── tools/
├── configs/
├── ckpts/
│   ├── pretrain_qformer/
│   ├── merge_trajproj.pth
│   ├── traj_queries.pt
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── conv/
│   │   ├── desc/
│   │   ├── keywords/
│   │   ├── vqa/
│   │   ├── nuscenes3d_occ_ego_temporal_infos_train_classv3.pkl
│   │   ├── nuscenes3d_occ_ego_temporal_infos_val_classv3.pkl
│   │   ├── kmeans_plan_36.npy
│   │   ├── sparsedrive_infos_nuscenes_infos_train.pkl
│   │   ├── ...
```

You can also refer to the official [OmniDrive](https://github.com/NVlabs/OmniDrive) documentation to supplement some QA-dataset-related folders.




