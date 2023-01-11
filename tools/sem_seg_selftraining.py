#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
import sys
import random
import logging
import os
from collections import OrderedDict
import torch
from torch import nn
import numpy as np
import time
import math
import PIL.Image as Image
from PIL import ImageFont
from PIL import ImageDraw
import datetime
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import gc
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg, add_hrnet_config
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    DatasetEvaluators,
    SemSegEvaluator,
    inference_on_dataset,
)
from detectron2.data.samplers import TrainingSampler, RandomClassSubsampling
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.generic_sem_seg_dataset import load_dataset_from_txt, load_dataset_to_inference, load_dataset_from_txt_and_merge
from contextlib import ExitStack, contextmanager
from cityscapesscripts.helpers.labels import trainId2label, labels
from detectron2.utils.logger import log_every_n_seconds
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

logger = logging.getLogger("detectron2")
DS_RATE = 4
softmax2d = nn.Softmax2d()
gt_path = '/datatmp/Datasets/segmentation/cityscapes/gtFine_trainvaltest/gtFine/train'
# gt_path = '/datatmp/Datasets/segmentation/Mapillary_Vistas/v2.0/training/labels'
# gt_path = '/datatmp/Datasets/segmentation/bdd100k/labels/masks/train'
gt_dataset = 'cityscapes'
dict_classes = {0:'Road', 1:'Sidewalk', 2:'Building', 3:'Wall', 4:'Fence', 5:'Pole', 6:'Traffic light', 7:'Traffic sign'
                , 8:'Vegetation', 9:'Terrain', 10:'Sky', 11:'Person', 12:'Rider', 13:'Car', 14:'Truck', 15:'Bus',
                16:'Train', 17:'Motorcycle', 18:'Bicycle', 19:'Void'}


def cotraining_argument_parser(parser):
    # Adds cotrainig arguments to the detectron2 base parser
    parser.add_argument(
        '--unlabeled_dataset_A',
        dest='unlabeled_dataset_A',
        help='File with Data A images',
        default=None,
        type=str
    )
    parser.add_argument(
        '--unlabeled_dataset_A_name',
        dest='unlabeled_dataset_A_name',
        help='Unlabeled dataset name to call dataloader function',
        default=None,
        type=str
    )    
    parser.add_argument(
        '--weights_branchA',
        dest='weights_branchA',
        help='Weights File of branch1',
        default=None,
        type=str
    )
    parser.add_argument(
        '--num-epochs',
        dest='epochs',
        help='Number of selftraining rounds',
        default=20,
        type=int
    )
    parser.add_argument(
        '--max_unlabeled_samples',
        dest='max_unlabeled_samples',
        help='Number of maximum unlabeled samples',
        default=500,
        type=int
    ) 
    parser.add_argument(
        '--samples',
        dest='samples',
        help='Number of top images to be sampled after each iteration',
        default=40,
        type=int
    )
    parser.add_argument(
        '--step_inc',
        dest='step_inc',
        help='Fix a image step to avoid consecutive images in secuences',
        default=1,
        type=int
    )
    parser.add_argument(
        '--continue_epoch',
        dest='continue_epoch',
        help='Continue co-training at the begining of the specified epoch',
        default=0,
        type=int
    )
    parser.add_argument(
        '--no_progress',
        action='store_true'
    )
    parser.add_argument(
        '--scratch_training',
        help='Use pretrained model for training in each epoch',
        action='store_true'
    )
    parser.add_argument(
        '--best_model',
        help='Use the best model obtained during the epochs',
        action='store_true'
    )
    parser.add_argument(
        '--initial_score_A',
        dest='initial_score_A',
        help='Initial score to reach to propagate weights to the next epoch',
        default=0,
        type=float
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set a prefixed seed to random select the unlabeled data. Useful to replicate experiments',
        default=None,
        type=int
    )
    parser.add_argument(
        '--mask_file',
        dest='mask_file',
        help='Mask file to apply to pseudolabels',
        default=None,
        type=str
    )
    parser.add_argument(
        '--recompute_all_pseudolabels',
        help='Use source B statistics to order samples by less confidence',
        action='store_true'
    )
    parser.add_argument(
        '--use_param_weights',
        help='Force the weights on config in a continue_training',
        action='store_true'
    )
    parser.add_argument(
        '--only_pseudolabeling',
        help='Compute 1 cycle of pseudolabels only',
        action='store_true'
    )
    parser.add_argument(
        '--no_random',
        help='No random selection of pseudolabels',
        action='store_true'
    )
    parser.add_argument(
        '--prior_file',
        dest='prior_file',
        help='Class prior file from source dataset to apply to the pseudolabels',
        default=None,
        type=str
    )
    parser.add_argument(
        '--weights_inference',
        dest='weights_inference',
        help='Initial weights to generate pseudolabels',
        default=None,
        type=str
    )
    return parser


def print_txt_format(results_dict, iter_name, epoch, output, model_id):
    with open(os.path.join(output,'results.txt'),"a+") as f:
        logger.info('----- Epoch: %s iteration: %s Model: %s -----' % (epoch,iter_name,model_id))
        f.write('----- Epoch: %s iteration: %s Model: %s ----- \n' % (epoch,iter_name,model_id))
        for k, v in results_dict['sem_seg'].items():
            if 'IoU' in k:
                logger.info('%s: %.4f' % (k, v))
                f.write('%s: %.4f \n' % (k, v))
        logger.info('\n')
        f.write('\n')


def built_custom_dataset(cfg, image_dir, gt_dir, dataset_name, add_pseudolabels=False, pseudo_img_dir=None,
                         pseudo_dir=None, test=False):
    global dict_classes
    if add_pseudolabels and pseudo_img_dir is not None and pseudo_dir is not None:
        DatasetCatalog.register(
                                dataset_name, lambda x1=image_dir, x2=pseudo_img_dir, y1=gt_dir,
                                y2=pseudo_dir: load_dataset_from_txt_and_merge(x1, x2, y1, y2,
                                                                               num_samples=cfg.DATASETS.TRAIN_SAMPLES)
                                )
    else:
        if test:
            DatasetCatalog.register(
                                    dataset_name, lambda x=image_dir, y=gt_dir: load_dataset_from_txt(x, y)
                                    )
        else:
            DatasetCatalog.register(
                                    dataset_name, lambda x=image_dir,
                                    y=gt_dir: load_dataset_from_txt(x, y, num_samples=cfg.DATASETS.TRAIN_SAMPLES)
                                    )
    if cfg.DATASETS.LABELS == 'cityscapes':
        MetadataCatalog.get(dataset_name).stuff_classes = [k.name for k in labels if 19 > k.trainId > -1]
        MetadataCatalog.get(dataset_name).stuff_colors = [k.color for k in labels if 19 > k.trainId > -1]
    elif cfg.DATASETS.LABELS == 'simple_TDA':
        dict_classes = {0:'Road', 1:'Vegetation', 2:'Sky', 3:'Fence', 4:'Pedestrian', 5:'Car', 6:'Background'}
        dict_color_map = {0:[255, 255, 153], 1:[107, 142, 35], 2:[31, 120, 180], 3:[106, 61, 154],
                      4:[123, 66, 173], 5:[115, 30, 218], 6:[0, 0, 0]}
        MetadataCatalog.get(dataset_name).stuff_classes = [dict_classes[k] for k in dict_classes]
        MetadataCatalog.get(dataset_name).stuff_colors = [dict_color_map[k] for k in dict_color_map]
    elif cfg.DATASETS.LABELS == 'simple_TDA_4c':
        dict_classes = {0:'Road', 1:'Pedestrian', 2:'Car', 3:'Background'}
        dict_color_map = {0:[255, 255, 153], 1:[123, 66, 173], 2:[115, 30, 218], 3:[0, 0, 0]}
        MetadataCatalog.get(dataset_name).stuff_classes = [dict_classes[k] for k in dict_classes]
        MetadataCatalog.get(dataset_name).stuff_colors = [dict_color_map[k] for k in dict_color_map]
    elif cfg.DATASETS.LABELS == 'simple_TDA_8c':
        dict_classes = {0:'Road', 1:'Vegetation', 2:'Sky', 3:'Fence', 4:'Pedestrian', 5:'Car', 6:'Sidewalk',
                        7:'Background'}
        dict_color_map = {0:[255, 255, 153], 1:[107, 142, 35], 2:[31, 120, 180], 3:[106, 61, 154],
                      4:[123, 66, 173], 5:[115, 30, 218], 6:[244, 35, 232], 7:[0, 0, 0]}
        MetadataCatalog.get(dataset_name).stuff_classes = [dict_classes[k] for k in dict_classes]
        MetadataCatalog.get(dataset_name).stuff_colors = [dict_color_map[k] for k in dict_color_map]
    elif cfg.DATASETS.LABELS == 'simple_TDA_9c':
        dict_classes = {0:'Road', 1:'Vegetation', 2:'Sky', 3:'Fence', 4:'Pedestrian', 5:'Car', 6:'Sidewalk',
                        7:'Background', 8:'Unpaved road'}
        dict_color_map = {0:[255, 255, 153], 1:[107, 142, 35], 2:[31, 120, 180], 3:[106, 61, 154],
                      4:[123, 66, 173], 5:[115, 30, 218], 6:[244, 35, 232], 7:[0, 0, 0], 8:[177, 89, 40]}
        MetadataCatalog.get(dataset_name).stuff_classes = [dict_classes[k] for k in dict_classes]
        MetadataCatalog.get(dataset_name).stuff_colors = [dict_color_map[k] for k in dict_color_map]
    else:
        raise Exception('Unsupported label set')
    MetadataCatalog.get(dataset_name).set(
                        image_dir=image_dir,
                        gt_dir=gt_dir,
                        evaluator_type="generic_sem_seg",
                        ignore_label=255,
                        )


def built_inference_dataset(im_list, dataset_name):
    DatasetCatalog.register(dataset_name, lambda x=im_list: load_dataset_to_inference(x))
    MetadataCatalog.get(dataset_name).set(
        image_dir=im_list,
        evaluator_type="generic_sem_seg",
        ignore_label=255,
    )


def build_sem_seg_train_aug(input, augmentation, void_label):
    augs = []
    if input.RESIZED:
        augs.append(T.Resize(input.RESIZE_SIZE))
    if input.ACTIVATE_MIN_SIZE_TRAIN:
        augs.append(T.ResizeShortestEdge(
            input.MIN_SIZE_TRAIN, input.MAX_SIZE_TRAIN, input.MIN_SIZE_TRAIN_SAMPLING))
    if input.CROP.ENABLED:
        if input.CROP.ENABLE_CLASS_ADAPTATIVE:
            augs.append(T.RandomCrop_ClassAdaptative(
                    input.CROP.TYPE,
                    input.CROP.SIZE,
                    input.CROP.CLASS_LIST))
        else:
            augs.append(T.RandomCrop_CategoryAreaConstraint(
                    input.CROP.TYPE,
                    input.CROP.SIZE,
                    input.CROP.SINGLE_CATEGORY_MAX_AREA,
                    void_label,
                    input.CROP.UPPER_MARGIN))
    if augmentation.HFLIP:
        augs.append(T.RandomFlip(prob=augmentation.HFLIP_PROB, horizontal=True, vertical=False))
    if augmentation.VFLIP:
        augs.append(T.RandomFlip(prob=augmentation.VFLIP_PROB, horizontal=False, vertical=True))
    if augmentation.CUTOUT:
        augs.append(T.CutOutPolicy(augmentation.CUTOUT_N_HOLES, augmentation.CUTOUT_LENGTH))
    if augmentation.RANDOM_RESIZE:
        augs.append(T.TrainScalePolicy(augmentation.RESIZE_RANGE))
    return augs

def build_sem_seg_train_aug2(input, augmentation, void_label):
    augs = []
    if input.RESIZED2:
        augs.append(T.Resize(input.RESIZE_SIZE2))
    if input.ACTIVATE_MIN_SIZE_TRAIN2:
        augs.append(T.ResizeShortestEdge(
            input.MIN_SIZE_TRAIN2, input.MAX_SIZE_TRAIN2, input.MIN_SIZE_TRAIN_SAMPLING2))
    if input.CROP2.ENABLED:
        if input.CROP2.ENABLE_CLASS_ADAPTATIVE:
            augs.append(T.RandomCrop_ClassAdaptative(
                    input.CROP2.TYPE,
                    input.CROP2.SIZE,
                    input.CROP2.CLASS_LIST))
        else:
            augs.append(T.RandomCrop_CategoryAreaConstraint(
                    input.CROP2.TYPE,
                    input.CROP2.SIZE,
                    input.CROP2.SINGLE_CATEGORY_MAX_AREA,
                    void_label,
                    input.CROP2.UPPER_MARGIN))
    if augmentation.HFLIP:
        augs.append(T.RandomFlip(prob=augmentation.HFLIP_PROB, horizontal=True, vertical=False))
    if augmentation.VFLIP:
        augs.append(T.RandomFlip(prob=augmentation.VFLIP_PROB, horizontal=False, vertical=True))
    if augmentation.CUTOUT:
        augs.append(T.CutOutPolicy(augmentation.CUTOUT_N_HOLES, augmentation.CUTOUT_LENGTH))
    if augmentation.RANDOM_RESIZE:
        augs.append(T.TrainScalePolicy(augmentation.RESIZE_RANGE))
    return augs


def build_sem_seg_pseudolabels_aug(input, augmentation, void_label):
    augs = []
    if input.RESIZED:
        augs.append(T.Resize(input.RESIZE_SIZE))
    if input.ACTIVATE_MIN_SIZE_TRAIN:
        augs.append(T.ResizeShortestEdge(
            input.MIN_SIZE_TRAIN, input.MAX_SIZE_TRAIN, input.MIN_SIZE_TRAIN_SAMPLING))
    if input.CROP.ENABLED:
        augs.append(T.RandomCrop_CategoryAreaConstraint(
                input.CROP.TYPE,
                input.CROP.SIZE,
                input.CROP.SINGLE_CATEGORY_MAX_AREA,
                void_label,
                input.CROP.UPPER_MARGIN))
    if augmentation.HFLIP:
        augs.append(T.RandomFlip(prob=augmentation.HFLIP_PROB, horizontal=True, vertical=False))
    if augmentation.VFLIP:
        augs.append(T.RandomFlip(prob=augmentation.VFLIP_PROB, horizontal=False, vertical=True))
    if augmentation.CUTOUT:
        augs.append(T.CutOutPolicy(augmentation.CUTOUT_N_HOLES, augmentation.CUTOUT_LENGTH))
    if augmentation.RANDOM_RESIZE:
        augs.append(T.TrainScalePolicy(augmentation.RESIZE_RANGE))
    return augs


def get_evaluator(cfg, args, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type == "generic_sem_seg":
        return SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder,
                               write_outputs=args.write_outputs, plot_transparency=args.plot_transparency,
                               write_conf_maps=args.write_conf_maps, ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                               void_metric=args.void_metric, num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
    if len(evaluator_list) == 0:
        raise NotImplementedError("no Evaluator for the dataset {} with the type {}".format(dataset_name,
                                                                                            evaluator_type))
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def plot_confusion_matrix(conf_matrix, epoch, iteration, branch, save_path):
    _, ax = plt.subplots(figsize=(25, 25))
    plt.rcParams.update({'font.size': 16})
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix = np.nan_to_num(conf_matrix)
    conf_matrix = np.around(conf_matrix, decimals=2)
    confd = ConfusionMatrixDisplay(conf_matrix)
    fig = confd.plot(cmap='Blues', ax=ax).figure_
    fig.suptitle('Confusion matrix epoch %s iteration %s branch %s' % (epoch, iteration, branch))
    fig.savefig(os.path.join(save_path, 'conf_matrix_epoch_%s_iter_%s_branch_%s.png' % (epoch, iteration, branch)))


def do_test_txt(cfg, args, model, dataset_name, step_iter, epoch, model_id):
    results = OrderedDict()
    if cfg.INPUT.VAL_RESIZE_SIZE is not None:
        mapper = DatasetMapper(cfg, is_train=False, augmentations=[T.Resize(cfg.INPUT.VAL_RESIZE_SIZE)])
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    else:
        data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = get_evaluator(cfg, args, dataset_name, None)
    results_i, conf_mat = inference_on_dataset(model, data_loader, evaluator, True)
    plot_confusion_matrix(conf_mat, epoch, step_iter, model_id, cfg.OUTPUT_DIR)
    results[dataset_name] = results_i
    print_txt_format(results_i, step_iter, epoch, cfg.OUTPUT_DIR, model_id)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def isolate_class_maps(cfg, output, img_list, pseudolabels_work_dir, idx):
    import matplotlib.pyplot as plt
    filename = img_list[idx][0].split('/')[-1].split('.')[0] + '.png'
    for cls in cfg.PSEUDOLABELING.ISOLATE_CLASSES:
        output_path = os.path.join(pseudolabels_work_dir, "class_heatmap", dict_classes[cls])
        create_folder(output_path)
        output = output.cpu().numpy()
        heatmap = output[cls, :, :]
        img = Image.open(img_list[idx][0])
        plt.imsave(os.path.join(output_path, filename), heatmap, cmap='viridis')
        map = Image.open(os.path.join(output_path, filename))
        heat_alpha = heatmap*255
        map.putalpha(0)
        map = np.asarray(map)
        map[:,:,3] = heat_alpha.astype(int)
        map = Image.fromarray(map)
        img.paste(map, (0, 0), map)
        img.save(os.path.join(output_path, filename))


def inference_on_imlist(cfg, model, weights, dataset_name, pseudolabels_work_dir, img_list=None, prior=None,
                        aug_inference=False, aug_path=None):
    # Following the same detectron2.evaluation.inference_on_dataset function
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(weights)
    if aug_inference:
        mapper = DatasetMapper(cfg, is_train=False,
                               augmentations=build_sem_seg_pseudolabels_aug(cfg.INPUT_PSEUDO, cfg.AUGMENTATION_A,
                                                                     cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE))
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    elif cfg.INPUT.VAL_RESIZE_SIZE is not None:
        mapper = DatasetMapper(cfg, is_train=False, augmentations=[T.Resize(cfg.INPUT.VAL_RESIZE_SIZE)])
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        model.apply_postprocess = False
    else:
        data_loader = build_detection_test_loader(cfg, dataset_name)
    total = len(data_loader)
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        outputs = []
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            batch_outputs = model(inputs)
            for b_id, output in enumerate(batch_outputs):
                # Saving indexes and values of maximums instead the 20 channels scores to save memory
                output = output['sem_seg']
                output = torch.unsqueeze(output, 0)
                output = softmax2d(output)
                output = torch.squeeze(output)
                if cfg.PSEUDOLABELING.ISOLATE_CLASSES is not None:
                    isolate_class_maps(cfg, output, img_list, pseudolabels_work_dir, idx)
                if prior is not None:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    aux = torch.from_numpy(prior).to(device)
                    output[:-1,:,:] = output[:-1,:,:]*0.75 + output[:-1,:,:]*aux[:-1,:,:]*0.25
                    conf = torch.amax(output, 0).cpu().numpy()
                else:
                    conf = torch.amax(output, 0).cpu().numpy()
                output_labels = torch.argmax(output, dim=0).to(torch.uint8).cpu().numpy()
                if aug_inference and aug_path is not None:
                    np_img = inputs[b_id]['image'].numpy()
                    plt_img = Image.fromarray(np.moveaxis(np_img, 0, -1))
                    filename = 'aug_' + inputs[b_id]['file_name'].split('/')[-1].split('.')[0] + '.png'
                    plt_img.save(os.path.join(aug_path, filename))
                    # colour_label(output_labels, os.path.join(aug_path, 'labels', filename))
                outputs.append([output_labels, conf])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
    return outputs


def do_train(cfg, args, model, weights, train_dataset_name, test_dataset_name, model_id, save_checkpoints_path, epoch,
             cls_thresh=None, resume=False, dataset_pseudolabels=None):
    model.train()
    model.apply_postprocess = True
    alternate_inputs = False
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, save_checkpoints_path, optimizer=optimizer, scheduler=scheduler
    )
    max_iter = cfg.SOLVER.MAX_ITER
    if resume:
        start_iter = (
                checkpointer.resume_or_load(weights, resume=resume).get("iteration", -1) + 1
            )
    else:
        checkpointer.resume_or_load(weights)

        start_iter = 0

    periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
        )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # Data aug mapper
    if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
        if cfg.INPUT.MIXED_DATA:
            mapper = DatasetMapper(cfg, is_train=True,
                                   augmentations=build_sem_seg_train_aug(cfg.INPUT, cfg.AUGMENTATION_A,
                                                                         cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE),
                                   dataset2_name=cfg.INPUT.DATASET_NAME,
                                   augmentations2=build_sem_seg_train_aug2(cfg.INPUT, cfg.AUGMENTATION_A,
                                                                           cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE))
        else:
            mapper = DatasetMapper(cfg, is_train=True,
                                   augmentations=build_sem_seg_train_aug(cfg.INPUT, cfg.AUGMENTATION_A,
                                                                         cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE))
        dataset: List[Dict] = DatasetCatalog.get(train_dataset_name)
        if cfg.DATASETS.TRAIN_IMG_TXT2 is not None and cfg.DATASETS.TRAIN_GT_TXT2 is not None:
            alternate_inputs = True
            if cfg.INPUT2.MIXED_DATA:
                mapper2 = DatasetMapper(cfg, is_train=True,
                                        augmentations=build_sem_seg_train_aug(cfg.INPUT2, cfg.AUGMENTATION,
                                                                              cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE),
                                        dataset2_name=cfg.INPUT2.DATASET_NAME,
                                        augmentations2=build_sem_seg_train_aug2(cfg.INPUT, cfg.AUGMENTATION,
                                                                                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE))
            else:
                mapper2 = DatasetMapper(cfg, is_train=True,
                                        augmentations=build_sem_seg_train_aug(cfg.INPUT2, cfg.AUGMENTATION,
                                                                              cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE))
            dataset2_A_source = cfg.DATASETS.TRAIN_NAME + '_A_source2' + str(epoch)
            built_custom_dataset(cfg, cfg.DATASETS.TRAIN_IMG_TXT2, cfg.DATASETS.TRAIN_GT_TXT2,
                                                       dataset2_A_source)
            dataset2: List[Dict] = DatasetCatalog.get(dataset2_A_source)
    else:
        mapper = None

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        if cfg.SOLVER.ALTERNATE_SOURCE_PSEUDOLABELS and dataset_pseudolabels is not None:
            dataset_pseudo: List[Dict] = DatasetCatalog.get(dataset_pseudolabels)
            mapper_pseudo = DatasetMapper(cfg, is_train=True,
                                          augmentations=build_sem_seg_train_aug(cfg.INPUT_PSEUDO, cfg.AUGMENTATION_A,
                                                                         cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE))
            data_loader_pseudo = build_detection_train_loader(cfg, dataset=dataset_pseudo, mapper=mapper_pseudo,
                                                        total_batch_size=cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1])
            if cfg.INPUT.RCS.ENABLED:
                sampler = RandomClassSubsampling(dataset, cfg, cfg.SOLVER.IMS_PER_BATCH)
                if alternate_inputs:
                    sampler2 = RandomClassSubsampling(dataset2, cfg, cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1])
            else:
                sampler = TrainingSampler(len(dataset))
                if alternate_inputs:
                    sampler2 = TrainingSampler(len(dataset2))
            if cfg.SOLVER.ACTIVATE_CLASSMIX:
                # If classmix is active we add "num pseudolables per batch" more samples to the source batch to use
                # them for classmix on the pseudolabels
                if alternate_inputs:
                    batch_size = int((cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0]
                                  + cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1])/2)
                    data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                               total_batch_size=batch_size)
                    data_loader2 = build_detection_train_loader(cfg, dataset=dataset2, mapper=mapper2, sampler=sampler2,
                                                                total_batch_size=batch_size)
                    results_list = training_loop_classmix_multidatasets(cfg, args, model, start_iter, max_iter, data_loader,
                                                                  data_loader2, data_loader_pseudo, storage, optimizer,
                                                                  scheduler, periodic_checkpointer, writers,
                                                                  test_dataset_name, epoch, model_id, cls_thresh)
                else:
                    data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                        total_batch_size=cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0]
                                                                        + cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1])
                    results_list = training_loop_classmixdatasets(cfg, args, model, start_iter, max_iter, data_loader,
                                                                  data_loader_pseudo, storage, optimizer, scheduler,
                                                                  periodic_checkpointer, writers, test_dataset_name,
                                                                  epoch, model_id, cls_thresh)
            else:
                if alternate_inputs:
                    batch_size = int(cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0]/2)
                    data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                               total_batch_size=batch_size)
                    data_loader2 = build_detection_train_loader(cfg, dataset=dataset2, mapper=mapper2, sampler=sampler2,
                                                                total_batch_size=batch_size)
                    results_list = training_loop_mix_multidatasets(cfg, args, model, start_iter, max_iter, data_loader,
                                                                   data_loader2, data_loader_pseudo, storage, optimizer,
                                                                   scheduler, periodic_checkpointer, writers,
                                                                   test_dataset_name, epoch, model_id)
                else:
                    data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                            total_batch_size=cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0])
                    results_list = training_loop_mixdatasets(cfg, args, model, start_iter, max_iter, data_loader,
                                                             data_loader_pseudo, storage, optimizer, scheduler,
                                                             periodic_checkpointer, writers, test_dataset_name, epoch,
                                                             model_id)
        else:
            dataset: List[Dict] = DatasetCatalog.get(train_dataset_name)
            if cfg.INPUT.RCS.ENABLED:
                sampler = RandomClassSubsampling(dataset, cfg, cfg.SOLVER.IMS_PER_BATCH)
            else:
                sampler = TrainingSampler(len(dataset))
            data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                       total_batch_size=cfg.SOLVER.IMS_PER_BATCH)
            results_list = training_loop(cfg, args, model, start_iter, max_iter, data_loader, storage, optimizer, scheduler,
                                         periodic_checkpointer, writers, test_dataset_name, epoch, model_id)
    return results_list


def training_loop(cfg, args, model, start_iter, max_iter, data_loader, storage, optimizer, scheduler, periodic_checkpointer,
                  writers, test_dataset_name, epoch, model_id):
    results_list = []
    for data, iteration in zip(data_loader, range(start_iter, max_iter)):
        storage.iter = iteration
        loss_dict = model(data)
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict
        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        if comm.is_main_process():
            storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
        scheduler.step()
        if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
            results = do_test_txt(cfg, args, model, test_dataset_name, iteration+1, epoch, model_id)
            results_list.append([results['sem_seg']['mIoU'], iteration])
            comm.synchronize()
        if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)
    return results_list


def training_loop_mixdatasets(cfg, args, model, start_iter, max_iter, data_loader, data_loader_pseudo, storage, optimizer,
                              scheduler, periodic_checkpointer, writers, test_dataset_name, epoch, model_id):
    # Training loop that mixes two dataloaders to compose the final batch with the proportion specified
    results_list = []
    for data1, data2, iteration in zip(data_loader, data_loader_pseudo, range(start_iter, max_iter)):
        storage.iter = iteration
        data = data1+data2
        loss_dict = model(data)
        del data
        gc.collect()
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict
        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        if comm.is_main_process():
            storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
        scheduler.step()
        if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
            results = do_test_txt(cfg, args, model, test_dataset_name, iteration+1, epoch, model_id)
            results_list.append([results['sem_seg']['mIoU'], iteration])
            comm.synchronize()
        if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)
    return results_list


def training_loop_mix_multidatasets(cfg, args, model, start_iter, max_iter, data_loader, data_loader2, data_loader_pseudo,
                                    storage, optimizer, scheduler, periodic_checkpointer, writers, test_dataset_name,
                                    epoch, model_id):
    # Training loop that mixes two source dataloaders and target to compose the final batch with the proportion
    # specified
    results_list = []
    for data1, data2, data_pseudo, iteration in zip(data_loader, data_loader2, data_loader_pseudo,
                                                    range(start_iter, max_iter)):
        storage.iter = iteration
        data = data1+data2+data_pseudo
        loss_dict = model(data)
        del data
        gc.collect()
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict
        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        if comm.is_main_process():
            storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
        scheduler.step()
        if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
            results = do_test_txt(cfg, args, model, test_dataset_name, iteration+1, epoch, model_id)
            results_list.append([results['sem_seg']['mIoU'], iteration])
            comm.synchronize()
        if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)
    return results_list


def compute_classmix(cfg, source, target, cls_thresh=None):
    assert len(source) == len(target)
    for idx, src_item in enumerate(source):
        if cfg.SOLVER.CLASSMIX_TGT_CLASSES is not None:
            cls_sel = torch.tensor(cfg.SOLVER.CLASSMIX_TGT_CLASSES, dtype=torch.int)
            idx_classes = torch.zeros(src_item['sem_seg'].shape)
            for cls in cls_sel:
                idx_classes = torch.logical_or(idx_classes, src_item['sem_seg'] == cls)
        else:
            classes = torch.unique(src_item['sem_seg'])
            if cls_thresh is not None:
                cls_aux = []
                cls_per_confidence = np.argsort(cls_thresh)
                for i in range(len(cls_per_confidence)):
                    if cls_per_confidence[i] in classes and cls_thresh[cls_per_confidence[i]] != 0:
                        cls_aux.append(cls_per_confidence[i])
                        if len(cls_aux) == int(len(classes)/2):
                            break
                cls_sel = []
                for cls in classes:
                    if cls not in cls_aux:
                        cls_sel.append(cls)
                cls_sel = torch.from_numpy(np.asarray(cls_sel))
            else:
                cls_sel = classes[torch.randperm(len(classes))][:int(len(classes)/2)]
            idx_classes = (src_item['sem_seg'][..., None] != cls_sel).all(-1)
        target[idx]['sem_seg'][idx_classes] = src_item['sem_seg'][idx_classes]
        target[idx]['image'][:, idx_classes] = src_item['image'][:, idx_classes]
    return target


def training_loop_classmixdatasets(cfg, args, model, start_iter, max_iter, data_loader, data_loader_pseudo, storage,
                                   optimizer, scheduler, periodic_checkpointer, writers, test_dataset_name, epoch,
                                   model_id, cls_thresh):
    # Training loop that mixes two dataloaders to compose the final batch with the proportion specified
    results_list = []
    for data1, data2, iteration in zip(data_loader, data_loader_pseudo, range(start_iter, max_iter)):
        pseudo_bz = len(data2)
        if cfg.SOLVER.CLASSMIX_ON_CLASS_DEMAND:
            data2 = compute_classmix(cfg, data1[-pseudo_bz:], data2, cls_thresh)
        else:
            data2 = compute_classmix(cfg, data1[-pseudo_bz:], data2)
        storage.iter = iteration
        data = data1[:-pseudo_bz]+data2
        loss_dict = model(data)
        del data
        gc.collect()
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict
        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        if comm.is_main_process():
            storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
        scheduler.step()
        if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
            results = do_test_txt(cfg, args, model, test_dataset_name, iteration+1, epoch, model_id)
            results_list.append([results['sem_seg']['mIoU'],iteration])
            # Compared to "train_net.py", the test results are not dumped to EventStorage
            comm.synchronize()
        if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)
    return results_list


def training_loop_classmix_multidatasets(cfg, args, model, start_iter, max_iter, data_loader, data_loader2,
                                         data_loader_pseudo, storage, optimizer, scheduler, periodic_checkpointer,
                                         writers, test_dataset_name, epoch, model_id, cls_thresh):
    # Training loop that mixes two dataloaders to compose the final batch with the proportion specified
    results_list = []
    for data1, data2, data_pseudo, iteration in zip(data_loader, data_loader2, data_loader_pseudo,
                                                    range(start_iter, max_iter)):
        pseudo_bz = int(len(data_pseudo)/2)
        if cfg.SOLVER.CLASSMIX_ON_CLASS_DEMAND:
            data_pseudo = compute_classmix(cfg, data1[-pseudo_bz:], data_pseudo[:pseudo_bz], cls_thresh)
            data_pseudo2 = compute_classmix(cfg, data2[-pseudo_bz:], data_pseudo[-pseudo_bz:], cls_thresh)
        else:
            data_pseudo = compute_classmix(cfg, data1[-pseudo_bz:], data_pseudo[:pseudo_bz])
            data_pseudo2 = compute_classmix(cfg, data2[-pseudo_bz:], data_pseudo[-pseudo_bz:])
        storage.iter = iteration
        data = data1[:-pseudo_bz]+data2[:-pseudo_bz]+data_pseudo+data_pseudo2
        loss_dict = model(data)
        del data
        gc.collect()
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict
        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        if comm.is_main_process():
            storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
        scheduler.step()
        if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
            results = do_test_txt(cfg, args, model, test_dataset_name, iteration+1, epoch, model_id)
            results_list.append([results['sem_seg']['mIoU'],iteration])
            # Compared to "train_net.py", the test results are not dumped to EventStorage
            comm.synchronize()
        if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)
    return results_list


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_hrnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.unlabeled_dataset_A is not None:
        cfg.DATASETS.UNLABELED_DATASET_A = args.unlabeled_dataset_A
    if args.weights_branchA is not None:
        cfg.MODEL.WEIGHTS_BRANCH_A = args.weights_branchA
    if args.max_unlabeled_samples is not None:
        cfg.DATASETS.MAX_UNLABELED_SAMPLES = args.max_unlabeled_samples
    #cfg.freeze()
    default_setup(
        cfg, args
    )
    return cfg


def get_unlabeled_data(unlabeled_dataset, step_inc, seed, samples):
    with open(unlabeled_dataset,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    im_list.sort()
    init_indx = random.randrange(0, step_inc)
    indx_sampled = np.asarray(range(init_indx, len(im_list), step_inc), dtype=int)
    im_list = np.asarray(im_list)[indx_sampled]
    random.seed(seed)
    if samples > -1:
        im_list = random.sample(im_list.tolist(), min(len(im_list), samples))
    else:
        im_list = im_list.tolist()
    return im_list


def compute_mtp_thresholds(pred_conf, pred_cls_num, tgt_portion, num_classes):
    threshold = []
    #hand_thres = [1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9] #[1.0, 0.9, 0.9, 1.0]
    for i in range(num_classes):
        x = pred_conf[pred_cls_num == i]
        if len(x) == 0:
            threshold.append(0)
            continue        
        x = np.sort(x)
        pixels = (len(x)/pred_conf.size)*100
        logger.info("Class %s, pixels %.3f %%, mean %.2f, std %.2f" % (dict_classes[i], pixels, np.mean(x), np.std(x)))
        if type(tgt_portion) == np.ndarray:
            thres = x[int(np.round(len(x)*(1-tgt_portion[i])))]
        else:
            thres = x[int(np.round(len(x)*(1-tgt_portion)))]
        #if thres > hand_thres[i]:
        #    thres = hand_thres[i]
        threshold.append(thres)
    threshold = np.array(threshold)
    threshold[threshold > 0.9] = 0.9
    threshold[threshold < 0.5] = 0.5
    return threshold


def apply_mpt(outputs, num_classes, tgt_num, tgt_portion, void_label, mask_file=None, prior=None, prior_thres=0):
    pred_cls_num = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.uint8)
    pred_conf = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.float32)
    for index, output in enumerate(outputs):
        pred_cls_num[index] = output[0]
        pred_conf[index] = output[1]
    thres = compute_mtp_thresholds(pred_conf, pred_cls_num, tgt_portion, num_classes)
    logger.info("MPT thresholds: {}".format(thres))
    pseudolabels = []
    pseudolabels_not_filtered = []
    scores_list = []
    for index in range(tgt_num):
        pseudolabels_not_filtered.append(pred_cls_num[index])
        label = pred_cls_num[index].copy()
        # Apply mask to the pseudolabel (useful to remove detection on prefixed void parts (e.g. ego vehicle))
        if mask_file is not None:
            mask = np.asarray(Image.open(mask_file).convert('L'), dtype=bool)
            label[mask] = void_label
        prob = pred_conf[index]
        for i in range(num_classes):
            if prior is not None and prior_thres > 0:
                prior_conf_mask = prior[i, :, :].copy()
                prior_conf_mask[prior[i, :, :] >= prior_thres] = 1.0
                prior_conf_mask[prior[i, :, :] < prior_thres] *= 1.0/prior_thres
                # aux = prob*0.85 + prob*prior[i,:,:]*0.15
                aux = prob*prior_conf_mask
                label[(aux <= thres[i])*(label == i)] = void_label  # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(aux <= thres[i])*(label == i)] = np.nan
            else:
                label[(prob <= thres[i])*(label == i)] = void_label  # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(prob <= thres[i])*(label == i)] = np.nan
        pseudolabels.append(label)
        # Compute image score using mean of the weighted confidence pixels values higher than the threshold cls_thresh
        classes_id, pixel_count = np.unique(label, return_counts=True)
        if void_label > num_classes:
            score = np.nanmean(prob[label != num_classes])
        else:
            score = np.nanmean(prob)
        # create aux array for scores and pixel per class count
        aux_scores = np.zeros((num_classes+1), dtype=np.float32)
        aux_scores[-1] = score
        for idx, class_id in enumerate(classes_id):
            if class_id < num_classes:
                aux_scores[class_id] = pixel_count[idx]
        if void_label > num_classes:
            # guarantee minimum of foreground in score system (5% on image)
            if (np.sum(aux_scores[:num_classes])/label.size)*100 < 5:
                aux_scores[-1] = 0
        scores_list.append(aux_scores)
    return np.asarray(pseudolabels), np.asarray(scores_list), np.asarray(pseudolabels_not_filtered), np.asarray(thres)


def apply_mpt_with_gt_correction(outputs, unlabeled_dataset, num_classes, tgt_num, tgt_portion, void_label,
                                 perc_correction=None, mask_file=None, prior=None, prior_thres=0):
    pred_cls_num = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.uint8)
    pred_conf = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.float32)
    for index, output in enumerate(outputs):
        pred_cls_num[index] = output[0]
        pred_conf[index] = output[1]
    thres = compute_mtp_thresholds(pred_conf, pred_cls_num, tgt_portion, num_classes)
    logger.info("MPT thresholds: {}".format(thres))
    pseudolabels = []
    pseudolabels_not_filtered = []
    scores_list = []
    use_perc = False
    if perc_correction is not None and perc_correction > 0:
        idx_list = random.sample(list(range(tgt_num)),int(tgt_num*perc_correction))
        idx_list.sort()
        use_perc = True
    for index in range(tgt_num):
        pseudolabels_not_filtered.append(pred_cls_num[index])
        label = pred_cls_num[index].copy()
        # Apply mask to the pseudolabel (useful to remove detection on prefixed void parts (e.g. ego vehicle))
        if mask_file is not None:
            mask = np.asarray(Image.open(mask_file).convert('L'), dtype=bool)
            label[mask] = void_label
        prob = pred_conf[index]
        if not use_perc or index in idx_list:
            if gt_dataset == 'cityscapes':
                label_name = unlabeled_dataset[index][0].split('/')[-1]
                city = label_name.split('_')[0]
                gt_filename = '_'.join(label_name.split('_')[:-1]) + '_gtFine_labelTrainIds.png'
                gt_file = os.path.join(gt_path, city, gt_filename)
            else:
                label_name = unlabeled_dataset[index][0].split('/')[-1].split('.')[0] + '.png'
                gt_file = os.path.join(gt_path, label_name)
            gt = Image.open(gt_file).convert('L')
            if label.shape[0] != gt.size[1] and label.shape[1] != gt.size[0]:
                gt = gt.resize((label.shape[1], label.shape[0]))
            gt = np.asarray(gt, dtype=np.uint8)
        for i in range(num_classes):
            if prior is not None and prior_thres > 0:
                prior_conf_mask = prior[i, :, :].copy()
                prior_conf_mask[prior[i, :, :] >= prior_thres] = 1.0
                prior_conf_mask[prior[i, :, :] < prior_thres] *= 1.0/prior_thres
                # aux = prob*0.85 + prob*prior[i,:,:]*0.15
                aux = prob*prior_conf_mask
                label[(aux <= thres[i])*(label == i)] = void_label  # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(aux <= thres[i])*(label == i)] = np.nan
            else:
                label[(prob <= thres[i])*(label == i)] = void_label  # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(prob <= thres[i])*(label == i)] = np.nan
            # gt correction
            if not use_perc or index in idx_list:
                label[(prob > thres[i])*(label == i)*(label != void_label)] = gt[(prob > thres[i])*(label == i)*(label != void_label)]
        label[(label == 255)] = void_label
        pseudolabels.append(label)
        # Compute image score using mean of the weighted confidence pixels values higher than the threshold cls_thresh
        classes_id, pixel_count = np.unique(label, return_counts=True)
        score = np.nanmean(prob)
        # create aux array for scores and pixel per class count
        aux_scores = np.zeros((num_classes+1), dtype=np.float32)
        aux_scores[-1] = score
        for idx, class_id in enumerate(classes_id):
            aux_scores[class_id] = pixel_count[idx]
        scores_list.append(aux_scores)
    return np.asarray(pseudolabels), np.asarray(scores_list), np.asarray(pseudolabels_not_filtered), np.asarray(thres)


def apply_mtp_gt_score(outputs, unlabeled_dataset, num_classes, tgt_num, tgt_portion, void_label, mask_file=None,
                       prior=None, prior_thres=0):
    pred_cls_num = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.uint8)
    pred_conf = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.float32)
    for index, output in enumerate(outputs):
        pred_cls_num[index] = output[0]
        pred_conf[index] = output[1]
    thres = compute_mtp_thresholds(pred_conf, pred_cls_num, tgt_portion, num_classes)
    logger.info("MPT thresholds: {}".format(thres))
    pseudolabels = []
    pseudolabels_not_filtered = []
    scores_list = []
    for index in range(tgt_num):
        pseudolabels_not_filtered.append(pred_cls_num[index])
        label = pred_cls_num[index].copy()
        # Apply mask to the pseudolabel (useful to remove detection on prefixed void parts (e.g. ego vehicle))
        if mask_file is not None:
            mask = np.asarray(Image.open(mask_file).convert('L'), dtype=bool)
            label[mask] = void_label
        prob = pred_conf[index]
        for i in range(num_classes):
            if prior is not None:
                prior_conf_mask = prior[i,:,:].copy()
                prior_conf_mask[prior[i,:,:] >= prior_thres] = 1.0
                prior_conf_mask[prior[i,:,:] < prior_thres] *= 1.0/prior_thres
                #aux = prob*0.85 + prob*prior[i,:,:]*0.15
                aux = prob*prior_conf_mask
                label[(aux<=thres[i])*(label==i)] = void_label  # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(aux<=thres[i])*(label==i)] = np.nan
            else:
                label[(prob<=thres[i])*(label==i)] = void_label  # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(prob<=thres[i])*(label==i)] = np.nan
        pseudolabels.append(label)
        label_name = unlabeled_dataset[index][0].split('/')[-1]
        city = label_name.split('_')[0]
        gt_filename = '_'.join(label_name.split('_')[:-1]) + '_gtFine_labelTrainIds.png'
        gt_file = os.path.join(gt_path,city,gt_filename)
        gt = Image.open(gt_file).convert('L')
        gt = np.asarray(gt, dtype=np.uint8)
        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        label64 = np.array(label, dtype=np.int64)
        conf_matrix = np.bincount(
            (num_classes) * label64.reshape(-1) + gt.reshape(-1),
            minlength=conf_matrix.size,
        ).reshape(conf_matrix.shape)
        conf_matrix = conf_matrix.astype('float') / conf_matrix[:,:num_classes-1].sum(axis=1)[:, np.newaxis]
        conf_matrix = np.nan_to_num(conf_matrix)
        conf_matrix = np.around(conf_matrix, decimals=2)
        tp = conf_matrix[:,:num_classes-1].diagonal()
        score = 0
        cls_num = 0
        for i in range(num_classes-1):
            if np.sum(conf_matrix[i,:num_classes-1]):
                cls_num += 1
                score += tp[i]
        score = score/cls_num
        # Compute image score using the mean of the weighted confidence pixels values higher than the threshold
        # cls_thresh
        classes_id, pixel_count = np.unique(label, return_counts=True)
        # create aux array for scores and pixel per class count
        aux_scores = np.zeros((num_classes+1), dtype=np.float32)
        aux_scores[-1] = score
        for idx, class_id in enumerate(classes_id):
            aux_scores[class_id] = pixel_count[idx]
        scores_list.append(aux_scores)
    return np.asarray(pseudolabels), np.asarray(scores_list), np.asarray(pseudolabels_not_filtered), np.asarray(thres)


def apply_mtp_and_gt_substraction(outputs, unlabeled_dataset, num_classes, tgt_num, tgt_portion, void_label,
                                  mask_file=None, prior=None, prior_thres=0, gt_dataset = ''):
    pred_cls_num = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.uint8)
    pred_conf = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.float32)
    for index, output in enumerate(outputs):
        pred_cls_num[index] = output[0]
        pred_conf[index] = output[1]
    thres = compute_mtp_thresholds(pred_conf, pred_cls_num, tgt_portion, num_classes)
    logger.info("MPT thresholds: {}".format(thres))
    pseudolabels = []
    pseudolabels_not_filtered = []
    scores_list = []
    gt_substraction_stats = []
    for index in range(tgt_num):
        pseudolabels_not_filtered.append(pred_cls_num[index])
        label = pred_cls_num[index].copy()
        # Apply mask to the pseudolabel (useful to remove detection on prefixed void parts (e.g. ego vehicle))
        if mask_file is not None:
            mask = np.asarray(Image.open(mask_file).convert('L'), dtype=bool)
            label[mask] = void_label
        prob = pred_conf[index]
        if gt_dataset == 'cityscapes':
            label_name = unlabeled_dataset[index][0].split('/')[-1]
            city = label_name.split('_')[0]
            gt_filename = '_'.join(label_name.split('_')[:-1]) + '_gtFine_labelTrainIds.png'
            gt_file = os.path.join(gt_path, city, gt_filename)
        else:
            label_name = unlabeled_dataset[index][0].split('/')[-1].split('.')[0] + '.png'
            gt_file = os.path.join(gt_path, label_name)
        gt = Image.open(gt_file).convert('L')
        if label.shape[0] != gt.size[1] and label.shape[1] != gt.size[0]:
            gt = gt.resize((label.shape[1], label.shape[0]))
        gt = np.asarray(gt, dtype=np.uint8)
        aux_scores_subs = np.zeros((num_classes*2), dtype=np.float32)
        for i in range(num_classes):
            if prior is not None and prior_thres > 0:
                prior_conf_mask = prior[i,:,:].copy()
                prior_conf_mask[prior[i,:,:] >= prior_thres] = 1.0
                prior_conf_mask[prior[i,:,:] < prior_thres] *= 1.0/prior_thres
                # aux = prob*0.85 + prob*prior[i,:,:]*0.15
                aux = prob*prior_conf_mask
                label[(aux<=thres[i])*(label==i)] = void_label # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(aux<=thres[i])*(label==i)] = np.nan
            else:
                label[(prob<=thres[i])*(label==i)] = void_label # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(prob<=thres[i])*(label==i)] = np.nan
        label_subs = label.copy()
        label_subs[label_subs!=gt] = void_label
        for i in range(num_classes):
            gt_pixels = np.sum(gt == i)
            label_pixels = np.sum(label == i)
            label_subs_pixels = np.sum(label_subs == i)
            if label_pixels > 0 and label_subs_pixels > 0:
                aux_scores_subs[2*i] = 1 - (label_subs_pixels/float(label_pixels))
                aux_scores_subs[2*i+1] = label_subs_pixels/float(gt_pixels)
        gt_substraction_stats.append(aux_scores_subs)
        pseudolabels.append(label_subs)
        # Compute image score using the mean of the weighted confidence pixels values higher than the threshold
        # cls_thresh
        classes_id, pixel_count = np.unique(label, return_counts=True)
        score = np.nanmean(prob)
        # create aux array for scores and pixel per class count
        aux_scores = np.zeros((num_classes+1), dtype=np.float32)
        aux_scores[-1] = score
        for idx, class_id in enumerate(classes_id):
            aux_scores[class_id] = pixel_count[idx]
        scores_list.append(aux_scores)
    gt_substraction_stats = np.asarray(gt_substraction_stats)
    std_subs = np.nanstd(np.where(gt_substraction_stats!=0,gt_substraction_stats,np.nan), axis=0)*100
    mean_subs = np.nanmean(np.where(gt_substraction_stats!=0,gt_substraction_stats,np.nan), axis=0)*100
    logger.info("--- MPT stats w.r.t GT ---")
    for i in range(num_classes-1):
        logger.info("%s: FP mean %.2f%%, std %.2f%% | TP mean %.2f%%, std %.2f%%" % (dict_classes[i], mean_subs[2*i],
                                                                                  std_subs[2*i], mean_subs[2*i+1],
                                                                                  std_subs[2*i+1]))
    return np.asarray(pseudolabels), np.asarray(scores_list), np.asarray(pseudolabels_not_filtered), \
        np.asarray(thres), gt_substraction_stats


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# Finds the best font size
def find_font_size(max_width, classes, font_file, max_font_size=100):

    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))

    # Find the maximum font size that all labels fit into the box width
    n_classes = len(classes)
    for c in range(n_classes):
        text = classes[c]
        for s in range(max_font_size, 1, -1):
            font = ImageFont.truetype(font_file, s)
            txt_size = draw.textsize(text, font=font)
            # print('c:{} s:{} txt_size:{}'.format(c, s, txt_size))
            if txt_size[0] <= max_width:
                max_font_size = s
                break

    # Find the maximum box height needed to fit the labels
    max_font_height = 1
    font = ImageFont.truetype(font_file, max_font_size)
    for c in range(n_classes):
        max_font_height = max(max_font_height,
                              draw.textsize(text, font=font)[1])

    return max_font_size, int(max_font_height)


# Draw class legend in an image
def draw_legend(w, color_map, classes, n_lines=3, txt_color=(255, 255, 255),
                font_file="Cicle_Gordita.ttf"):

    # Compute legend sizes
    n_classes = len(color_map)
    n_classes_per_line = int(math.ceil(float(n_classes) / n_lines))
    class_width = int(w/n_classes_per_line)
    font_size, class_height = find_font_size(class_width, classes, font_file)
    font = ImageFont.truetype(font_file, font_size)

    # Create PIL image
    img_pil = Image.new('RGB', (w, n_lines*class_height))
    draw = ImageDraw.Draw(img_pil)

    # Draw legend
    for i in range(n_classes):
        # Get color and label
        color = color_map[i]
        text = classes[i]

        # Compute current row and col
        row = int(i/n_classes_per_line)
        col = int(i % n_classes_per_line)

        # Draw box
        box_pos = [class_width*col, class_height*row,
                   class_width*(col+1), class_height*(row+1)]
        draw.rectangle(box_pos, fill=color, outline=None)

        # Draw text
        txt_size = draw.textsize(text, font=font)[0]
        txt_pos = [box_pos[0]+((box_pos[2]-box_pos[0])-txt_size)/2, box_pos[1]]
        draw.text(txt_pos, text, txt_color, font=font)

    return np.asarray(img_pil)


def colour_label(inference, filename, dataset_name=None):
    pred_colour = 255 * np.ones([inference.shape[0],inference.shape[1],3], dtype=np.uint8)
    color_map = []
    classes = []
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        class_names = meta.stuff_classes
        stuff_colors = meta.stuff_colors
        for idx in range(len(stuff_colors)):
            pred_colour[(inference == idx),0] = stuff_colors[idx][0]
            pred_colour[(inference == idx),1] = stuff_colors[idx][1]
            pred_colour[(inference == idx),2] = stuff_colors[idx][2]
            color_map.append(tuple(stuff_colors[idx]))
            classes.append(class_names[idx])
    else:
        for train_id, label in trainId2label.items():
            pred_colour[(inference == train_id),0] = label.color[0]
            pred_colour[(inference == train_id),1] = label.color[1]
            pred_colour[(inference == train_id),2] = label.color[2]
        for i in range(len(trainId2label)):
            if i in trainId2label:
                color_map.append(trainId2label[i].color)
                classes.append(trainId2label[i].name)
    legend = draw_legend(inference.shape[1], color_map, classes, n_lines=2)
    pred_colour = np.concatenate((pred_colour, legend))
    Image.fromarray(pred_colour).save(filename)


def save_pseudolabels(num_classes, images, pseudolabels, scores, pseudolabels_path, coloured_pseudolabels_path=None,
                      pseudolabels_not_filtered=None, coloured_pseudolabels_not_filtered_path=None, file_text='',
                      images_aug_path=None, dataset_name=None):
    filenames_and_scores = os.path.join('/'.join(pseudolabels_path.split('/')[:-1]),'%sfilenames_and_scores.txt' %
                                        (file_text))
    images_txt = os.path.join('/'.join(pseudolabels_path.split('/')[:-1]),'%sselected_images_path.txt' %
                              (file_text))
    psedolabels_txt = os.path.join('/'.join(pseudolabels_path.split('/')[:-1]),'%sselected_pseudolabels_path.txt' %
                                   (file_text))
    with open(filenames_and_scores,'w') as f:
        with open(images_txt,'w') as g:
            with open(psedolabels_txt,'w') as h:
                for idx, image in enumerate(images):
                    if images_aug_path is not None:
                        filename = 'aug_' + image[0].split('/')[-1].split('.')[0] + '.png'
                    else:
                        filename = image[0].split('/')[-1].split('.')[0] + '.png'
                    Image.fromarray(pseudolabels[idx]).save(os.path.join(pseudolabels_path,filename))
                    if coloured_pseudolabels_path is not None:
                        colour_label(pseudolabels[idx], os.path.join(coloured_pseudolabels_path,filename), dataset_name)
                    if pseudolabels_not_filtered is not None and coloured_pseudolabels_not_filtered_path is not None:
                        colour_label(pseudolabels_not_filtered[idx],
                                     os.path.join(coloured_pseudolabels_not_filtered_path, filename), dataset_name)
                    # Create txt with files names and scores
                    if images_aug_path is not None:
                        g.write('%s\n' % (os.path.join(images_aug_path, filename)))
                    else:
                        g.write('%s\n' % (image[0]))
                    f.write('%s %s %s %s\n' % (filename, str(scores[idx][-1]), str(scores[idx][-2]),
                                               str(np.count_nonzero(scores[idx][:num_classes-1]))))
                    h.write('%s\n' % (os.path.join(pseudolabels_path,filename)))
    return images_txt, psedolabels_txt, filenames_and_scores


def merge_txts_and_save(new_txt, txt1, txt2=None):
    if txt2 is not None:
        files = [txt1, txt2]
    else:
        files = [txt1]
    with open(new_txt, 'w') as f:
        for file in files:
            with open(file) as infile:
                for line in infile:
                    f.write(line)
    return new_txt


def update_best_score_txts_and_save(accum_scores_txt, accum_images_txt, accum_labels_txt, new_scores_txt, 
                                    new_images_txt, new_labels_txt, save_img_txt, save_labels_txt, save_scores_txt,
                                    sorting_method):
    with open(accum_scores_txt,'r') as f:
        accum_scores = [line.rstrip().split(' ') for line in f.readlines()]
    with open(new_scores_txt,'r') as f:
        new_scores_txt = [line.rstrip().split(' ') for line in f.readlines()]
    with open(accum_images_txt,'r') as f:
        accum_images = [line.rstrip().split(' ') for line in f.readlines()]
    with open(new_images_txt,'r') as f:
        new_images = [line.rstrip().split(' ') for line in f.readlines()]
    with open(accum_labels_txt,'r') as f:
        accum_labels = [line.rstrip().split(' ') for line in f.readlines()]
    with open(new_labels_txt,'r') as f:
        new_labels = [line.rstrip().split(' ') for line in f.readlines()]
    ignore_list = []
    # Check for repeated images
    for idx, score in enumerate(new_scores_txt):
        for idx2, score2 in enumerate(accum_scores):
            if score[0] == score2[0]:
                # Depending of the sorting method we use scores or number of void pixel to update
                if sorting_method == 'per_class' or sorting_method == 'per_void_pixels':
                    check = score[2] < score2[2]
                else:
                    check = score[1] > score2[1]
                if check:
                    # If we found the same image with better score we updated values in all the acumulated lists
                    accum_scores[idx2][1] = score[1]
                    accum_scores[idx2][2] = score[2]
                    accum_scores[idx2][3] = score[3]
                    accum_labels[idx2] = new_labels[idx]
                # we store the index to do not add it again later
                ignore_list.append(idx)
                break
    # add new images into the accumulated ones
    for idx, score in enumerate(new_scores_txt):
        if idx not in ignore_list:
            accum_scores.append(score)
            accum_labels.append(new_labels[idx])
            accum_images.append(new_images[idx])
    # save each data in its respective txt
    new_img_dataset = open(save_img_txt,'w')
    new_labels_dataset = open(save_labels_txt,'w')
    new_scores_dataset = open(save_scores_txt,'w')
    for idx, _ in enumerate(accum_scores):
        new_img_dataset.write(accum_images[idx][0] + '\n')
        new_labels_dataset.write(accum_labels[idx][0] + '\n')
        new_scores_dataset.write(accum_scores[idx][0] + ' ' + accum_scores[idx][1] + ' ' + accum_scores[idx][2] + ' '
                                 + accum_scores[idx][3] + '\n')
    new_img_dataset.close()
    new_labels_dataset.close()
    new_scores_dataset.close()
    return save_img_txt, save_labels_txt, save_scores_txt


def sorting_scores(scores, sorting_method, selftraining=False):
    if sorting_method == 'per_class':
        sorted_idx = np.lexsort((scores[:,-1],np.count_nonzero(scores[:,:-2], axis=1)))[::-1]
    elif sorting_method == 'per_void_pixels':
        # Sorting by number of void pixels (lower to higher)
        sorted_idx = np.argsort(scores[:,-2])
    elif sorting_method == 'confidence':
        # Sorting by confidence (lower to higher for cotraining)
        sorted_idx = np.argsort(scores[:,-1])
        if selftraining:
            # (higher to lower for selftraining)
            sorted_idx = sorted_idx[::-1][:len(scores)]
    else:
        #No sorting
        sorted_idx = np.arange(len(scores))
    return sorted_idx


def get_data(data_list):
    with open(data_list,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    return im_list


def custom_thresholding(outputs, unlabeled_dataset, tgt_num, npz_path):
    # Option 1: use only joan thresholds
    # Option 2: avg of inference confidence and joan
    for index, _ in enumerate(outputs):
        print(outputs[index][1])
        file_name = unlabeled_dataset[index][0].split('/')[-1].split('.')[0]
        npz_file = os.path.join(npz_path, file_name + '.npz')
        npz = np.load(open(npz_file,'rb'))
        confidence = npz['confidence']
        outputs[index][1] = confidence
        print(outputs[index][1])
    return outputs


def create_pseudolabels_folders(pseudolabels_work_dir):
    pseudolabels_path_model = os.path.join(pseudolabels_work_dir,'pseudolabels')
    create_folder(pseudolabels_path_model)
    coloured_pseudolabels_path_model = os.path.join(pseudolabels_work_dir,'coloured_pseudolabels')
    create_folder(coloured_pseudolabels_path_model)
    coloured_pseudolabels_not_filtered_path_model = os.path.join(pseudolabels_work_dir,
                                                                   'coloured_pseudolabels_not_filtered')
    create_folder(coloured_pseudolabels_not_filtered_path_model)
    return pseudolabels_path_model, coloured_pseudolabels_path_model, coloured_pseudolabels_not_filtered_path_model


def sort_pseudolabels(cfg, pseudolabels, unlabeled_dataset, pseudolabels_not_filtered, scores_list,
                      pseudolabels_path, coloured_pseudolabels_path, coloured_pseudolabels_not_filtered_path,
                      aug_image_path=None, dataset_name=None):
    logger.info("Sorting mode: {}".format(cfg.PSEUDOLABELING.SORTING))
    num_selected = cfg.PSEUDOLABELING.NUMBER
    sorted_idx = sorting_scores(scores_list, cfg.PSEUDOLABELING.SORTING, selftraining=True)
    sorted_scores_listA = scores_list[sorted_idx]
    sorted_pseudolabels_A = pseudolabels[sorted_idx]
    sorted_unlabeled_datasetA = unlabeled_dataset[sorted_idx]
    sorted_pseudolabels_A_not_filtered = pseudolabels_not_filtered[sorted_idx]
    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

    logger.info("Select candidates and Save on disk")
    # select candidates and save them to add them to the source data
    if len(sorted_unlabeled_datasetA) > cfg.PSEUDOLABELING.NUMBER:
        images_txt, psedolabels_txt, filenames_and_scores = \
            save_pseudolabels(num_classes, sorted_unlabeled_datasetA[:num_selected], sorted_pseudolabels_A[:num_selected],
                              sorted_scores_listA[:num_selected], pseudolabels_path,
                              coloured_pseudolabels_path, sorted_pseudolabels_A_not_filtered[:num_selected],
                              coloured_pseudolabels_not_filtered_path, images_aug_path=aug_image_path,
                              dataset_name=dataset_name)
    else:
        images_txt, psedolabels_txt, filenames_and_scores = \
            save_pseudolabels(num_classes, sorted_unlabeled_datasetA, sorted_pseudolabels_A, sorted_scores_listA,
                              pseudolabels_path, coloured_pseudolabels_path,
                              sorted_pseudolabels_A_not_filtered,
                              coloured_pseudolabels_not_filtered_path, images_aug_path=aug_image_path,
                              dataset_name=dataset_name)

    return images_txt, psedolabels_txt, filenames_and_scores


def compute_accumulation(cfg, accumulation_mode, dataset_path, images_txt, accumulated_selection_img, pseudolabels_txt,
                         accumulated_selection_pseudo, filenames_and_scores, accumulated_scores):
    logger.info("Compute data accumulation procedure selected: {}".format(accumulation_mode))
    if accumulation_mode is not None and len(accumulated_selection_img) > 0:
        if accumulation_mode.lower() == 'all':
            accumulated_selection_img = merge_txts_and_save(os.path.join(dataset_path,'dataset_img.txt'),
                                                            accumulated_selection_img,
                                                            images_txt)
            accumulated_selection_pseudo = merge_txts_and_save(os.path.join(dataset_path,'dataset_pseudolabels.txt'),
                                                               accumulated_selection_pseudo,
                                                               pseudolabels_txt)
            accumulated_scores = merge_txts_and_save(os.path.join(dataset_path,'filenames_and_scores.txt'),
                                                     accumulated_scores,
                                                     filenames_and_scores)
        if accumulation_mode.lower() == 'update_best_score':
            accumulated_selection_img, accumulated_selection_pseudo, accumulated_scores = \
                update_best_score_txts_and_save(
                                            accumulated_scores, accumulated_selection_img,
                                            accumulated_selection_pseudo,
                                            filenames_and_scores,
                                            images_txt,
                                            pseudolabels_txt,
                                            os.path.join(dataset_path,'dataset_img.txt'),
                                            os.path.join(dataset_path,'dataset_pseudolabels.txt'),
                                            os.path.join(dataset_path,'filenames_and_scores.txt'),
                                            cfg.PSEUDOLABELING.SORTING)
    else:
        # No accumulation, only training with new pseudolabels
        accumulated_selection_img = merge_txts_and_save(os.path.join(dataset_path,'dataset_img.txt'),
                                                                images_txt)
        accumulated_selection_pseudo = merge_txts_and_save(os.path.join(dataset_path,
                                                                         'dataset_pseudolabels.txt'),
                                                                pseudolabels_txt)
        accumulated_scores = merge_txts_and_save(os.path.join(dataset_path,'filenames_and_scores.txt'),
                                                                filenames_and_scores)

    return accumulated_selection_img, accumulated_selection_pseudo, accumulated_scores

def main(args):
    cfg = setup(args)
    continue_epoch = args.continue_epoch
    pseudolabeling = cfg.PSEUDOLABELING.MODE
    accumulated_selection_imgA = []
    accumulated_selection_pseudoA = []
    accumulated_scores_A = []
    accumulation_mode = cfg.PSEUDOLABELING.ACCUMULATION
    data_aug_pseudolabels = cfg.INPUT_PSEUDO.DATA_AUG
    void_label = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    if args.weights_inference is not None:
        weights_inference_branchA = args.weights_inference
    else:
        weights_inference_branchA = cfg.MODEL.WEIGHTS_BRANCH_A
    weights_train_branchA = cfg.MODEL.WEIGHTS_BRANCH_A
    tgt_portion = cfg.PSEUDOLABELING.INIT_TGT_PORT
    if type(tgt_portion) == list:
        tgt_portion = np.asarray(tgt_portion)
        max_list_tgt = np.fmin(tgt_portion + cfg.PSEUDOLABELING.MAX_TGT_PORT, 1.0)
    # Set initial scores to surpass during an epoch to propagate weghts to the next one
    best_score_A = args.initial_score_A

    # Build test dataset
    built_custom_dataset(cfg, cfg.DATASETS.TEST_IMG_TXT, cfg.DATASETS.TEST_GT_TXT, cfg.DATASETS.TEST_NAME, test=True)

    # set a seed for the unlabeled data selection
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randrange(sys.maxsize)

    if args.prior_file is not None:
        source_priors = np.load(args.prior_file)
    else:
        source_priors = None
    prior_thres=0.1
    prior_relax=0.05

    # Start self-training
    for epoch in range(args.continue_epoch,args.epochs):
        if continue_epoch > 0 and not args.only_pseudolabeling and not args.use_param_weights:
            weights_inference_branchA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch-1),'checkpoints/model_final.pth')
            if type(tgt_portion) == np.ndarray:
                tgt_portion = np.where(tgt_portion >= max_list_tgt,
                                        max_list_tgt,
                                        tgt_portion + cfg.PSEUDOLABELING.TGT_PORT_STEP*continue_epoch)
            else:
                tgt_portion = min(tgt_portion + cfg.PSEUDOLABELING.TGT_PORT_STEP*continue_epoch,
                                  cfg.PSEUDOLABELING.MAX_TGT_PORT)
            prior_thres = max(prior_thres-(prior_relax*continue_epoch), 0)

        pseudolabels_work_dir = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'pseudolabeling')
        pseudolabels_path, coloured_pseudolabels_path, coloured_pseudolabels_not_filtered_path =\
            create_pseudolabels_folders(pseudolabels_work_dir)
        if data_aug_pseudolabels:
            pseudolabels_aug_work_dir = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'pseudolabeling_aug')
            images_aug_path = os.path.join(pseudolabels_aug_work_dir, 'images')
            create_folder(images_aug_path)
            pseudolabels_aug_path, coloured_pseudolabels_aug_path, coloured_pseudolabels_aug_not_filtered_path =\
                create_pseudolabels_folders(pseudolabels_aug_work_dir)
        dataset_A_path = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'unlabeled_data_selected')
        create_folder(dataset_A_path)
        checkpoints_A_path = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'checkpoints')
        create_folder(checkpoints_A_path)

        logger.info("Starting training from iteration {}".format(epoch))
        # prepare unlabeled data
        logger.info("prepare unlabeled data")
        seed = seed + epoch
        logger.info("Seed for unlabeled data {}".format(seed))
        if args.no_random:
            unlabeled_datasetA = get_data(cfg.DATASETS.UNLABELED_DATASET_A)
        else:
            # Read unlabeled data from the txt specified and select randomly X samples defined on max_unlabeled_samples
            unlabeled_datasetA = get_unlabeled_data(cfg.DATASETS.UNLABELED_DATASET_A, args.step_inc, seed,
                                                    cfg.DATASETS.MAX_UNLABELED_SAMPLES)
        logger.info("Unlabeled data selected from {}: {}".format(cfg.DATASETS.UNLABELED_DATASET_A,
                                                                 len(unlabeled_datasetA)))
        # Regiter unlabeled dataset on detectron 2
        built_inference_dataset(unlabeled_datasetA, args.unlabeled_dataset_A_name)
        # Compute inference on unlabeled datasets
        #cfg.INPUT.CROP.ENABLED = False
        model = build_model(cfg)
        logger.info("Compute inference on unlabeled datasets")
        start_time = time.perf_counter()
        # Inference return a tuple of labels and confidences
        inference_A = inference_on_imlist(cfg, model, weights_inference_branchA, args.unlabeled_dataset_A_name,
                                          pseudolabels_work_dir, unlabeled_datasetA)
        if data_aug_pseudolabels:
            inference_aug = inference_on_imlist(cfg, model, weights_inference_branchA, args.unlabeled_dataset_A_name,
                                                pseudolabels_aug_work_dir, unlabeled_datasetA, aug_inference=True,
                                                aug_path=images_aug_path)

        total_time = time.perf_counter() - start_time
        logger.info("Compute inference on unlabeled dataset A: {:.2f} s".format(total_time))
        logger.info("Pseudolabeling mode: {}, Threshold: {}".format(pseudolabeling, tgt_portion))
        if pseudolabeling == 'mpt':
            start_time = time.perf_counter()
            pseudolabels_A, scores_listA, pseudolabels_A_not_filtered, cls_thresh = \
                apply_mpt(inference_A, num_classes, len(unlabeled_datasetA),
                          tgt_portion, void_label, args.mask_file, source_priors, prior_thres)
            if data_aug_pseudolabels:
                pseudolabels_aug, scores_list_aug, pseudolabels_aug_not_filtered, cls_thresh_aug = \
                    apply_mpt(inference_aug, num_classes, len(unlabeled_datasetA),
                              tgt_portion)
            total_time = time.perf_counter() - start_time
            logger.info("MPT on unlabeled dataset A: {:.2f} s".format(total_time))
        elif pseudolabeling == 'mpt_gt_substraction':
            start_time = time.perf_counter()
            pseudolabels_A, scores_listA, pseudolabels_A_not_filtered, cls_thresh, gt_substraction_stats = \
                apply_mtp_and_gt_substraction(inference_A, unlabeled_datasetA, num_classes,
                                              len(unlabeled_datasetA), tgt_portion, void_label,
                                              args.mask_file, gt_dataset=gt_dataset)
            total_time = time.perf_counter() - start_time
            logger.info("MPT on unlabeled dataset A: {:.2f} s".format(total_time))
        elif pseudolabeling == 'mpt_gt_score':
            start_time = time.perf_counter()
            pseudolabels_A, scores_listA, pseudolabels_A_not_filtered, cls_thresh = \
                apply_mtp_gt_score(inference_A, unlabeled_datasetA, num_classes,
                                   len(unlabeled_datasetA), tgt_portion, void_label, args.mask_file)
            total_time = time.perf_counter() - start_time
            logger.info("MPT on unlabeled dataset A: {:.2f} s".format(total_time))
        elif pseudolabeling == 'mpt_gt_correction':
            start_time = time.perf_counter()

            pseudolabels_A, scores_listA, pseudolabels_A_not_filtered, cls_thresh = \
                apply_mpt_with_gt_correction(inference_A, unlabeled_datasetA, num_classes,
                                             len(unlabeled_datasetA), tgt_portion, void_label, mask_file=args.mask_file,
                                             perc_correction=0.2)
            total_time = time.perf_counter() - start_time
            logger.info("MPT on unlabeled dataset A: {:.2f} s".format(total_time))
        else:
            raise Exception('unknown pseudolabeling method defined')

        # Continue cotraining on the specified epoch
        if continue_epoch > 0:
            accumulated_selection_imgA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch-1),
                                                      'unlabeled_data_selected/dataset_img.txt')
            accumulated_selection_pseudoA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch-1),
                                                         'unlabeled_data_selected/dataset_pseudolabels.txt')
            accumulated_scores_A = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch-1),
                                                'unlabeled_data_selected/filenames_and_scores.txt')
            continue_epoch = 0

        # save pseudolabels
        # Order pseudolabels by method selected on config file
        unlabeled_datasetA = np.asarray(unlabeled_datasetA)
        images_txt_A, psedolabels_txt_A, filenames_and_scoresA = sort_pseudolabels(cfg, pseudolabels_A,
                      unlabeled_datasetA, pseudolabels_A_not_filtered, scores_listA,
                      pseudolabels_path, coloured_pseudolabels_path, coloured_pseudolabels_not_filtered_path,
                                                                                   dataset_name=cfg.DATASETS.TEST_NAME)
        if data_aug_pseudolabels:
            images_txt_aug, psedolabels_txt_aug, filenames_and_scores_aug = sort_pseudolabels(cfg, pseudolabels_aug,
                      unlabeled_datasetA, pseudolabels_aug_not_filtered, scores_list_aug,
                      pseudolabels_aug_path, coloured_pseudolabels_aug_path,
                      coloured_pseudolabels_aug_not_filtered_path, aug_image_path=images_aug_path,
                                                                                    dataset_name=cfg.DATASETS.TEST_NAME)

        if not args.only_pseudolabeling:
            # Compute data accumulation procedure
            logger.info("Compute data accumulation procedure selected: {}".format(accumulation_mode))
            accumulated_selection_imgA, accumulated_selection_pseudoA, accumulated_scores_A = compute_accumulation(cfg,
                         accumulation_mode, dataset_A_path, images_txt_A, accumulated_selection_imgA, psedolabels_txt_A,
                         accumulated_selection_pseudoA, filenames_and_scoresA, accumulated_scores_A)
            if data_aug_pseudolabels:
                accumulated_selection_imgA, accumulated_selection_pseudoA, accumulated_scores_A = compute_accumulation(cfg,
                             accumulation_mode, dataset_A_path, images_txt_aug, accumulated_selection_imgA, psedolabels_txt_aug,
                             accumulated_selection_pseudoA, filenames_and_scores_aug, accumulated_scores_A)

            # Training step
            #cfg.INPUT.CROP.ENABLED = True
            #model = build_model(cfg)
            if cfg.SOLVER.ALTERNATE_SOURCE_PSEUDOLABELS:
                # create one dataloader for the source data and another for target pseudolabels
                dataset_A_source = cfg.DATASETS.TRAIN_NAME + '_A_source' + str(epoch)
                dataset_A_target = cfg.DATASETS.TRAIN_NAME + '_A_target' + str(epoch)
                built_custom_dataset(cfg, cfg.DATASETS.TRAIN_IMG_TXT, cfg.DATASETS.TRAIN_GT_TXT, dataset_A_source)
                built_custom_dataset(cfg, accumulated_selection_imgA, accumulated_selection_pseudoA, dataset_A_target)
                # Train model A
                logger.info("Training Model A")
                results_A = do_train(cfg, args, model, weights_train_branchA, dataset_A_source, cfg.DATASETS.TEST_NAME,'a',
                                     checkpoints_A_path, epoch, cls_thresh, resume=False,
                                     dataset_pseudolabels=dataset_A_target)

            else:
                # create dataloader adding psedolabels to source dataset
                dataset_A_name = cfg.DATASETS.TRAIN_NAME + '_A_' + str(epoch)
                built_custom_dataset(cfg, cfg.DATASETS.TRAIN_IMG_TXT, cfg.DATASETS.TRAIN_GT_TXT, dataset_A_name, True,
                                     accumulated_selection_imgA, accumulated_selection_pseudoA)
                # Train model A
                logger.info("Training Model A")
                results_A = do_train(cfg, args, model, weights_train_branchA, dataset_A_name, cfg.DATASETS.TEST_NAME,'a',
                                     checkpoints_A_path, epoch, args.continue_epoch, resume=False)

            # refresh weight file pointers after iteration for initial inference if there is improvement
            if args.best_model:
                # Assign best model obtained until now to generate the pseudolabels in the next cycle
                # Model only used for inference
                for score, iteration in results_A:
                    if score > best_score_A:
                        best_score_A = score
                        weights_inference_branchA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),
                                                                 'checkpoints/model_%s.pth' % (str(iteration).zfill(7)))
                        if not args.scratch_training:
                            weights_train_branchA = weights_inference_branchA
                logger.info("Best model A until now: {}".format(weights_inference_branchA))
                logger.info("Best mIoU: {}".format(best_score_A))
            else:
                if not args.no_progress:
                    # The model for the next inference and training cycle is the last one obtained
                    weights_inference_branchA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),
                                                             'checkpoints/model_final.pth')
                    if not args.scratch_training:
                        weights_train_branchA = weights_inference_branchA

            if epoch < 4 and args.recompute_all_pseudolabels and cfg.SOLVER.ALTERNATE_SOURCE_PSEUDOLABELS:
                re_pseudolabels_path_model_A = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),
                                                            'pseudolabeling/recomputed_pseudolabels')
                create_folder(re_pseudolabels_path_model_A)
                re_coloured_pseudolabels_pathA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),
                                                              'pseudolabeling/recomputed_coloured_pseudolabels')
                create_folder(re_coloured_pseudolabels_pathA)
                logger.info("Recompute accumulated pseudolabels and update")
                accumulated_imgA = get_data(accumulated_selection_imgA)
                accumulated_pseudoA = get_data(accumulated_selection_pseudoA)
                start_time = time.perf_counter()
                # Inference return a tuple of labels and confidences
                inference_A = inference_on_imlist(cfg, model, weights_inference_branchA, dataset_A_target,
                                                  pseudolabels_work_dir, accumulated_imgA)
                total_time = time.perf_counter() - start_time
                logger.info("Compute inference on unlabeled dataset A: {:.2f} s".format(total_time))
                logger.info("Pseudolabeling mode: {}".format(cfg.PSEUDOLABELING.MODE))
                if cfg.PSEUDOLABELING.MODE == 'mpt':
                    start_time = time.perf_counter()
                    pseudolabels_A, scores_listA, pseudolabels_A_not_filtered, cls_thresh = \
                        apply_mpt(inference_A, num_classes,len(accumulated_pseudoA), tgt_portion,
                                  void_label, args.mask_file, source_priors, prior_thres)
                    total_time = time.perf_counter() - start_time
                    logger.info("MPT on unlabeled dataset A: {:.2f} s".format(total_time))
                elif pseudolabeling == 'mpt_gt_correction':
                    start_time = time.perf_counter()
                    pseudolabels_A, scores_listA, pseudolabels_A_not_filtered, cls_thresh = \
                        apply_mpt_with_gt_correction(inference_A, unlabeled_datasetA, num_classes,
                                                     len(unlabeled_datasetA), tgt_portion, mask_file=args.mask_file,
                                                     perc_correction=0.2)
                    total_time = time.perf_counter() - start_time
                    logger.info("MPT on unlabeled dataset A: {:.2f} s".format(total_time))
                else:
                    raise Exception('unknown pseudolabeling method defined')

                # select candidates and save them to add them to the source data
                images_txt_A, psedolabels_txt_A, filenames_and_scoresA = save_pseudolabels(num_classes, accumulated_imgA,
                    pseudolabels_A, scores_listA, re_pseudolabels_path_model_A, re_coloured_pseudolabels_pathA,
                    file_text='recomp_', dataset_name=cfg.DATASETS.TEST_NAME)
                _, _, _ = update_best_score_txts_and_save(
                                                    accumulated_scores_A, accumulated_selection_imgA,
                                                    accumulated_selection_pseudoA,
                                                    filenames_and_scoresA, images_txt_A, psedolabels_txt_A,
                                                    os.path.join(dataset_A_path,'dataset_img.txt'),
                                                    os.path.join(dataset_A_path,'dataset_pseudolabels.txt'),
                                                    os.path.join(dataset_A_path,'filenames_and_scores.txt'),
                                                    cfg.PSEUDOLABELING.SORTING)

                 # free memory
                del inference_A
                del scores_listA
                del pseudolabels_A
                del accumulated_pseudoA
                del accumulated_imgA
                gc.collect()

            # Update thesholdings
            if type(tgt_portion) == np.ndarray:
                tgt_portion = np.where(tgt_portion >= max_list_tgt, max_list_tgt,
                                       tgt_portion + cfg.PSEUDOLABELING.TGT_PORT_STEP)
            else:
                tgt_portion = min(tgt_portion + cfg.PSEUDOLABELING.TGT_PORT_STEP, cfg.PSEUDOLABELING.MAX_TGT_PORT)
            prior_thres = max(prior_thres-prior_relax, 0)

            # delete all datasets registered during epoch
            DatasetCatalog.remove(args.unlabeled_dataset_A_name)
            MetadataCatalog.remove(args.unlabeled_dataset_A_name)
            if cfg.SOLVER.ALTERNATE_SOURCE_PSEUDOLABELS:
                DatasetCatalog.remove(dataset_A_source)
                MetadataCatalog.remove(dataset_A_source)
                DatasetCatalog.remove(dataset_A_target)
                MetadataCatalog.remove(dataset_A_target)
        else:
            return


if __name__ == "__main__":
    default_parser = default_argument_parser()
    args = cotraining_argument_parser(default_parser).parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
