
# Co-Training for Unsupervised Domain Adaptation of Semantic Segmentation Models

By Jose L. Gómez, Gabriel Villalonga and Antonio M. López

**[[Paper]](https://www.mdpi.com/1424-8220/23/2/621)**

**Abstract:** Semantic image segmentation is a core task for autonomous driving, which is performed by deep models. Since training these models draws to a curse of human-based image labeling, the use of synthetic images with automatically generated labels together with unlabeled real-world images is a promising alternative. This implies addressing an unsupervised domain adaptation (UDA) problem. In this paper, we propose a new co-training procedure for synth-to-real UDA of semantic segmentation models. It performs iterations where the (unlabeled) real-world training images are labeled by intermediate deep models trained with both the (labeled) synthetic images and the real-world ones labeled in previous iterations. More specifically, a self-training stage  provides two domain-adapted models and a model collaboration loop allows the mutual improvement of these two models. The final semantic segmentation labels (pseudo-labels) for the real-world images are provided by these two models. The overall procedure treats the deep models as black boxes and drives their collaboration at the level of pseudo-labeled target images, i.e., neither modifying loss functions is required, nor explicit feature alignment. We test our proposal on standard synthetic and real-world datasets for onboard semantic segmentation. Our procedure shows improvements ranging from approximately 13 to 31 mIoU points over baselines.

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

| Step          | Source         | Target     | mIoU  | Config file                              | Weights   |
|---------------|----------------|------------|-------|------------------------------------------|-----------|
| Baseline      | GTAV           | Cityscapes | 37.86 | configs/gtaV/baseline.yaml               | [model](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EcYC1FyZgC9IvxStnMNxm0YByc5bs7Dsw4C86qx1I4jK7Q?e=08CHnQ)  |
| Baseline + CB | GTAV           | Cityscapes | 42.76 | configs/gtaV/baseline+CB.yaml            | [model](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EVmCyaylLn9Htg9KbrJ0j6IB5_JV2txlsHKYy0YCOg4Wsw?e=CUWGGG)  |
| Self-training | GTAV           | Cityscapes | 53.49 | configs/gtaV/self_training.yaml          | [1](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EeGcU2Cg4aJIgTVQA85U1c4Bo57d5hKYKrH4moXgHqpLaQ?e=aB5cOb)[10](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EUtsuG5D8ZVGjbVx5skgov8B95IxP-k7LmUrGwikK0SB3w?e=5c0YVU)  |
| Co-training   | GTAV           | Cityscapes | 59.57 | configs/gtaV/final_step.yaml             | [A](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/Edyp7cRYlqNFlISn5UNurgcBKHfLCYETDaylYADc63nd7g?e=tukGIA)[B](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EYFZV0ShFP9Fg-etsG94f1ABPNsk9XUwaRIOjSeuKiD4Fg?e=lMCwD4)[Final](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EXLEJt9oHZhGqWBdbKJqXLABF0cTAcLadw1NbSCGJw01Bw?e=KkJJWA)  |
| Baseline      | Synscapes      | Cityscapes | 45.98 | configs/synscapes/baseline.yaml          | [model](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EVavfwJ3Vh5PsU8LyEsgCf0Bp2kbHgcp6vs4ojAFAMqIog?e=jFLnYp)  |
| Self-training | Synscapes      | Cityscapes | 55.34 | configs/synscapes/self_training.yaml     | [10](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EWSUouFEbqhBrwraDjKDmzwBm5g1NB4432_C-HYp-tdA9Q?e=J8H1pu)  |
| Co-training   | Synscapes      | Cityscapes | 58.38 | configs/synscapes/final_step.yaml        | [A](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EX19oat_XSJDnG0iMOzXcKMB-Tq5fgREZiSpFahonfq7kg?e=kAcH0Q)[B](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EYYGR9vKUolMpok79PZ7_JgBh2NTwYk1t8d97X46_Cq6dA?e=hqtytH)[Final](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EQ7yi63GAKtEnfjZHTfeNy8B2gnNcO-11QN3RWLmCZX-7A?e=lqWSdH)  |
| Baseline      | SYNTHIA        | Cityscapes | 39.48 | configs/SYNTHIA/baseline.yaml            | [model](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/ERwQujqlHNtGody32teb26MBxh9EQko3KM9YAz_RCXLKMw?e=tpdEVw)  |
| Self-training | SYNTHIA        | Cityscapes | 48.74 | configs/SYNTHIA/self_training.yaml       | [10](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EYlj5f0QqG1AodT_IbaPaPUBQibhlD8jraVfLWundWLEcA?e=fhpvEu)  |
| Co-training   | SYNTHIA        | Cityscapes | 56.09 | configs/SYNTHIA/final_step.yaml          | [A](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EccbK__5YvZHpEYtSbdaGhgB1ZTSQ9cOyi-0w27frnOorg?e=Oyf2h6)[B](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EX7kjtoLG_lIqbYONMteVvsBvkIHIfOSHl_sa4MCKZWSug?e=GwCe7W)[Final](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/Ee8VcbUJTBtLiICRuQMrCGoBWeWbfrtlUeyeO3lgWjtinw?e=4pGTzt)  |
| Baseline      | GTAV+Synscapes | Cityscapes | 59.32 | configs/gtaV+synscapes/baseline.yaml     | [model](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EY3z4dTUVtZAiUQVKnsLYeABnCAuSomtpBLacDlyKOBbJQ?e=YV5jWO)  |
| Self-training | GTAV+Synscapes | Cityscapes | 67.47 | configs/gtaV+synscapes/self_training.yaml| [1](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EXDxy256b1BFoqzMQGdUydUBkOeZ8WbGKUd0C0db5fQEBQ?e=ZWjb7v)[10](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EWSUouFEbqhBrwraDjKDmzwBm5g1NB4432_C-HYp-tdA9Q?e=VrmyKa)  |
| Co-training   | GTAV+Synscapes | Cityscapes | 70.23 | configs/gtaV+synscapes/final_step.yaml   | [A](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EaivyfCim4JNsL-MOTjGHfIBBbCWDXtBxp5wDxJe3aLO-g?e=Clo649)[B](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/ERC58x9z2BBFtujiBYI91-cBNxL5rXMYKDyKHMUybjK1ZQ?e=fli3UA)[Final](https://cvcuab-my.sharepoint.com/:u:/g/personal/jlgomez_cvc_uab_cat/EaBie7kXSdtDgiRc8NIElhIB9bMzyNg-Gol6e2f_0UHb6w?e=pcDXKF)  |

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
