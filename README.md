
# Co-Training for Unsupervised Domain Adaptation of Semantic Segmentation Models

By Jose L. Gómez, Gabriel Villalonga and Antonio M. López

**[[Paper]](https://www.mdpi.com/1424-8220/23/2/621)**

## Introduction

Welcome to the repository of our paper "Co-Training for Unsupervised Domain Adaptation of Semantic Segmentation Models". 
The code uses a modified forked version of the [Detectron2](https://github.com/facebookresearch/detectron2) framework. 
Inside the tools folder you can find the Co-training, Self-training and others implementations used on the paper to achieve the 
results reported.

## Requirements

- Linux with Python ≥ 3.6
- PyTorch ≥ 1.6 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
- OpenCV
- Numpy 
- PIL
- Cityscapes scripts included in Detectron2
- ./tools/packages_list.txt includes all the libraries and versions used to run the code (not all libraries are needed)
See [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- Hardware: NVIDIA GPU ≥12GB to reproduce the paper results.


## Setup

0. Installation:
**Clone** this github repository:
```bash
  git clone https://github.com/JoseLGomez/Co-training_SemSeg_UDA.git
  python -m pip install -e Co-training_SemSeg_UDA
  cd Co-training_SemSeg_UDA
```
Note: To rebuild detectron2 that’s built from a local clone, use ```rm -rf build/ **/*.so``` to clean the old build first. You often need to rebuild detectron2 after reinstalling PyTorch.

1. Datasets:
- Download [GTA-5](https://download.visinf.tu-darmstadt.de/data/from_games/)
- Download [Synscapes](https://synscapes.on.liu.se/download.html).
- Download [Cityscapes](https://www.cityscapes-dataset.com/).
- Download [BDD100K](https://bdd-data.berkeley.edu/).
- Download [Mapillary Vistas](https://www.cityscapes-dataset.com/).

2. List RGB images and GT:
- Create a .txt file with the images and their respective paths:

```
images.txt
/path/to/image/1.png
/path/to/image/2.png
/path/to/image/3.png
...
```
- Create a .txt file with the ground truth and their respective paths:

```
gt.txt
/path/to/ground/truth/1.png
/path/to/ground/truth/2.png
/path/to/ground/truth/3.png
...
```

Note: This can be done in linux using the command ls.
```
ls -1 $PWD/path/to/image/*.png > images.txt
ls -1 $PWD/path/to/ground/truth/*.png > gt.txt
```
- Ensure that the entries of both txt files match. In detail, the image from the first line on images.txt have its respective 
 ground truth in the first line on gt.txt and so on.

3. Offline LAB translation:
The colour correction in the LAB space explained in our paper is done offline running the script ```tool/apply_LAB_to_source_data.py```
- You need to set the variables ```source``` and ```target``` inside the code:
  - Set ```source=Path/to/synthetic/images.txt``` where images.txt is a synthetic dataset from step 2 (e.g. Synscapes, GTAV)
  - Set ```source=Path/to/real/images.txt``` where images.txt is a real dataset from step 2 (e.g. Cityscapes, Mapillary, BDD100k)
  - Set ```n_workers``` accordingly to the number of threads available in your machine (recommended 8).
- The script generate automatically the translated RGB images inside the source dataset folder on a new folder named ```rgb_translated```
- Update Step 2 

## Training [Only 1 GPU]

The training step is divided in four parts:

1. Baselines [with LAB translation]
The initial semantic segmentation models are trained using ```tools/train_net_progress.py``` with a config file that 
contains the hyper-parameters ```configs/X/baseline.yaml```, where X denotes the source dataset
```
CUDA_VISIBLE_DEVICES=0 python tools/train_net_progress.py 
--num-gpus 1 
--config-file /path/to/config/file.yaml
--write-outputs
OUTPUT_DIR /path/to/save/experiment
DATASETS.TRAIN_IMG_TXT /path/to/train/data/images.txt
DATASETS.TRAIN_GT_TXT /path/to/train/data/gt.txt
DATASETS.TEST_IMG_TXT /path/to/evaluation/data/images.txt
DATASETS.TEST_GT_TXT /path/to/evaluation/data/gt.txt
```
Note: uppercase variables (e.g. OUTPUT_DIR, DATASETS.TRAIN_IMG_TXT) overrides the values inside the config file
during the execution, without modify the config file. You can set these values inside the config file if desired.
Note2: if you do not want intermediate inferences remove argument ```--write-outputs```

2. Self-training step
Self-training step specified in the paper is done running the script ```tools/sem_seg_selftraining.py``` with its 
respective configuration file ```configs/X/self_training.yaml```
```
CUDA_VISIBLE_DEVICES=0 python tools/sem_seg_selftraining.py 
--num-gpus 1 
--config-file /path/to/config/file.yaml
--unlabeled_dataset_A /path/to/real/data/image.txt
--weights_branchA /path/to/baseline/weights.pth 
--unlabeled_dataset_A_name dataset_A 
--max_unlabeled_samples 500 
--scratch_training 
--num-epochs 10 
--seed 100 
--recompute_all_pseudolabels
--mask_file /tools/ego_vehicle_mask.png
OUTPUT_DIR /path/to/save/experiment
```

3. Co-training step
Co-training step specified in the paper is done running the script ```tools/sem_seg_cotrainingV3.py``` with its 
respective configuration file ```configs/X/co_training.yaml```
```
CUDA_VISIBLE_DEVICES=0 python tools/sem_seg_cotrainingV3.py 
--num-gpus 1 
--config-file /path/to/config/file.yaml
--unlabeled_dataset_A /path/to/real/data/image.txt
--same_domain 
--weights_branchA /path/to/self-training/0/weights.pth 
--weights_branchB /path/to/self-training/9/weights.pth 
--unlabeled_dataset_A_name dataset_A 
--unlabeled_dataset_B_name dataset_B 
--max_unlabeled_samples 500 
--num-epochs 5  
--min_pixels 5000 
--scratch_training 
--seed 100 
--recompute_all_pseudolabels 
--ensembles 
--mask_file /tools/ego_vehicle_mask.png 
OUTPUT_DIR /path/to/save/experiment
```

4. Final training step
- First, you need to generate the final pseudolabels of the target dataset using the next script
```
CUDA_VISIBLE_DEVICES=0 python sem_seg_cotrainingV3.py 
--num-gpus 1 
--config-file ../configs/sem_seg_cotraining/cotraining/sem_seg_cotraining_deeplabV3plus_gta+synscapes_translated_bdd.yaml 
--unlabeled_dataset_A /data/121-1/Datasets/segmentation/bdd100k/train_bdd10k_subset_clean_rgb.txt 
--same_domain 
--weights_branchA /data/121-2/Experiments/jlgomez/cotraining_sem_seg/Cotraining_bdd_V3plus_S+G_Best_confidence_MPT0.9_0.5_tgt0.5_200_8k_batch4_seed100_maskfile_ClassMixV2_self0_9/model_A/4/checkpoints/model_final.pth 
--weights_branchB /data/121-2/Experiments/jlgomez/cotraining_sem_seg/Cotraining_bdd_V3plus_S+G_Best_confidence_MPT0.9_0.5_tgt0.5_200_8k_batch4_seed100_maskfile_ClassMixV2_self0_9/model_B/4/checkpoints/model_final.pth 
--unlabeled_dataset_A_name dataset_A 
--unlabeled_dataset_B_name dataset_B 
--max_unlabeled_samples 500
--num-epochs 5  
--seed 100 
--ensembles 
--mpt_ensemble 
--no_training 
OUTPUT_DIR /path/to/save/experiment
DATASETS.TEST_IMG_TXT /path/to/target/training/set/images.txt
```
- Next, use the script of step 1 with the config file set to train simultaneously in batch time source data and 
pseudolabels ```configs/X/final_step.yaml```:
```
CUDA_VISIBLE_DEVICES=0 python tools/train_net_progress.py 
--num-gpus 1 
--config-file /path/to/config/file.yaml
--write-outputs
OUTPUT_DIR /path/to/save/experiment
```

## Evaluation
You can evaluate any model using the next script in addition with the config file containing the target dataset to evaluate
```
CUDA_VISIBLE_DEVICES=0 python tools/train_net_progress.py 
--num-gpus 1 
--config-file /path/to/config/file.yaml
--write-outputs
--eval-only
MODEL.WEIGHTS /path/to/model/weights.pth
OUTPUT_DIR /path/to/save/experiment
```

## Experimental results [DeeplabV3+]

| Step          | Source         | Target     | mIoU  | Config file                              | Weights |
|---------------|----------------|------------|-------|------------------------------------------|---------|
| Baseline      | GTAV           | Cityscapes | 37.86 | configs/gtaV/baseline.yaml               |
| Baseline + CB | GTAV           | Cityscapes | 42.76 | configs/gtaV/baseline+CB.yaml            |
| Self-training | GTAV           | Cityscapes | 53.49 | configs/gtaV/self_training.yaml          |
| Co-training   | GTAV           | Cityscapes | 59.57 | configs/gtaV/final_step.yaml             |
|---------------|----------------|------------|-------|------------------------------------------|
| Baseline      | Synscapes      | Cityscapes | 45.98 | configs/synscapes/baseline.yaml          |
| Self-training | Synscapes      | Cityscapes | 55.34 | configs/synscapes/self_training.yaml     |
| Co-training   | Synscapes      | Cityscapes | 58.38 | configs/synscapes/final_step.yaml        |
|---------------|----------------|------------|-------|------------------------------------------|
| Baseline      | SYNTHIA        | Cityscapes | 39.48 | configs/SYNTHIA/baseline.yaml            |
| Self-training | SYNTHIA        | Cityscapes | 48.74 | configs/SYNTHIA/self_training.yaml       |
| Co-training   | SYNTHIA        | Cityscapes | 56.09 | configs/SYNTHIA/final_step.yaml          |
|---------------|----------------|------------|-------|------------------------------------------|
| Baseline      | GTAV+Synscapes | Cityscapes | 59.32 | configs/gtaV+synscapes/baseline.yaml     |
| Self-training | GTAV+Synscapes | Cityscapes | 67.47 | configs/gtaV+synscapes/self_training.yaml|
| Co-training   | GTAV+Synscapes | Cityscapes | 70.23 | configs/gtaV+synscapes/final_step.yaml   |

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citation
Cite our work as follows:

```BibTeX
@Article{Gomez:2023,
AUTHOR = {Gómez, Jose L. and Villalonga, Gabriel and López, Antonio M.},
TITLE = {Co-Training for Unsupervised Domain Adaptation of Semantic Segmentation Models},
JOURNAL = {Sensors},
VOLUME = {23},
YEAR = {2023},
NUMBER = {2},
ARTICLE-NUMBER = {621}
}
```

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [Detectron2](https://github.com/facebookresearch/detectron2)
