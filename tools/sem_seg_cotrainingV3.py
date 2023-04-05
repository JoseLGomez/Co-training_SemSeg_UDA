import sys
import random
import logging
import os
from collections import OrderedDict
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
from detectron2.config import get_cfg
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
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.samplers import TrainingSampler, RandomClassSubsampling
from detectron2.data.datasets.generic_sem_seg_dataset import load_dataset_from_txt, load_dataset_to_inference, load_dataset_from_txt_and_merge
from torch import nn
import torch
from contextlib import ExitStack, contextmanager
from cityscapesscripts.helpers.labels import trainId2label, labels
from detectron2.utils.logger import log_every_n_seconds
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

logger = logging.getLogger("detectron2")
softmax2d = nn.Softmax2d()

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
        '--unlabeled_dataset_B',
        dest='unlabeled_dataset_B',
        help='File with Data B images',
        default=None,
        type=str
    )
    parser.add_argument(
        '--unlabeled_dataset_B_name',
        dest='unlabeled_dataset_B_name',
        help='Unlabeled dataset name to call dataloader function',
        default=None,
        type=str
    )
    parser.add_argument(
        '--same_domain',
        help='Set when the unlabeled domain for Data A and Data B is the same (i.e. rgb and mirrored rgb)',
        action='store_true'
    )
    parser.add_argument(
        '--weights_branchA',
        dest='weights_branchA',
        help='Weights File of branch1',
        default=None,
        type=str
    )
    parser.add_argument(
        '--weights_branchB',
        dest='weights_branchB',
        help='Weights File of branch2',
        default=None,
        type=str
    )
    parser.add_argument(
        '--num-epochs',
        dest='epochs',
        help='Number of cotraining iterations',
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
        '--scratch_training',
        help='Use pretrained model for training in each epoch',
        action='store_true'
    )
    parser.add_argument(
        '--ensembles',
        help='Generate pseudolabel with ensemble of the branches',
        action='store_true'
    )
    parser.add_argument(
        '--no_training',
        help='Skip training process',
        action='store_true'
    )
    parser.add_argument(
        '--fp_annot',
        action='store_true'
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set a prefixed seed to random select the unlabeled data. Useful to replicate experiments',
        default=None,
        type=int
    )
    parser.add_argument(
        '--min_pixels',
        dest='min_pixels',
        help='Minim number of pixels to filter a class on statistics',
        default=0,
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
        '--prior_file',
        dest='prior_file',
        help='Class prior file from source dataset to apply to the pseudolabels',
        default=None,
        type=str
    )
    parser.add_argument(
        '--ensembles_subtraction',
        help='Mode subtraction on cotraining ensemble',
        action='store_true'
    )
    parser.add_argument(
        '--mpt_ensemble',
        help='Use mtp on cotraining ensemble',
        action='store_true'
    )
    parser.add_argument(
        '--thres_A',
        dest='thres_A',
        help='Thresholds model A computer during co-training (used to generate final pseudolabels manually)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--thres_B',
        dest='thres_B',
        help='Thresholds model B computer during co-training (used to generate final pseudolabels manually)',
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
    else:
        raise Exception('Unsupported label set')
    MetadataCatalog.get(dataset_name).set(
                        image_dir=image_dir,
                        gt_dir=gt_dir,
                        evaluator_type="generic_sem_seg",
                        ignore_label=255,
                        )


def built_inference_dataset(cfg, im_list, dataset_name):
    DatasetCatalog.register(
        dataset_name, lambda x=im_list: load_dataset_to_inference(x)
    )
    MetadataCatalog.get(dataset_name).set(
        image_dir=im_list,
        evaluator_type="generic_sem_seg",
        ignore_label=255,
    )


def build_sem_seg_train_aug(input, augmentation, void_label):
    augs = []
    if input.ACTIVATE_MIN_SIZE_TRAIN:
        augs.append(T.ResizeShortestEdge(
            input.MIN_SIZE_TRAIN, input.MAX_SIZE_TRAIN, input.MIN_SIZE_TRAIN_SAMPLING))
    if input.RESIZED:
        augs.append(T.Resize(input.RESIZE_SIZE))
    if input.CROP.ENABLED:
        augs.append(T.RandomCrop_CategoryAreaConstraint(
                input.CROP.TYPE,
                input.CROP.SIZE,
                input.CROP.SINGLE_CATEGORY_MAX_AREA,
                void_label))
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
    if input.ACTIVATE_MIN_SIZE_TRAIN2:
        augs.append(T.ResizeShortestEdge(
            input.MIN_SIZE_TRAIN2, input.MAX_SIZE_TRAIN2, input.MIN_SIZE_TRAIN_SAMPLING2))
    if input.RESIZED2:
        augs.append(T.Resize(input.RESIZE_SIZE2))
    if input.CROP2.ENABLED:
        augs.append(T.RandomCrop_CategoryAreaConstraint(
                input.CROP2.TYPE,
                input.CROP2.SIZE,
                input.CROP2.SINGLE_CATEGORY_MAX_AREA,
                void_label))
    if augmentation.HFLIP:
        augs.append(T.RandomFlip(prob=augmentation.HFLIP_PROB, horizontal=True, vertical=False))
    if augmentation.VFLIP:
        augs.append(T.RandomFlip(prob=augmentation.VFLIP_PROB, horizontal=False, vertical=True))
    if augmentation.CUTOUT:
        augs.append(T.CutOutPolicy(augmentation.CUTOUT_N_HOLES, augmentation.CUTOUT_LENGTH))
    if augmentation.RANDOM_RESIZE:
        augs.append(T.TrainScalePolicy(augmentation.RESIZE_RANGE))
    return augs


def get_evaluator(cfg, dataset_name, output_folder=None, void_metric=False):
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
        '''return SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder, write_outputs=False,
                               val_resize=cfg.INPUT.VAL_RESIZE_SIZE)'''
        return SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder,
                               write_outputs=False, plot_transparency=False,
                               ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                               void_metric=void_metric, num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def plot_confusion_matrix(conf_matrix, epoch, iteration, branch, save_path):
    _, ax = plt.subplots(figsize=(25,25))
    plt.rcParams.update({'font.size': 16})
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix = np.nan_to_num(conf_matrix)
    conf_matrix = np.around(conf_matrix, decimals=2)
    confd = ConfusionMatrixDisplay(conf_matrix)
    fig = confd.plot(cmap='Blues', ax=ax).figure_
    fig.suptitle('Confusion matrix epoch %s iteration %s branch %s' % (epoch, iteration, branch))
    fig.savefig(os.path.join(save_path,'conf_matrix_epoch_%s_iter_%s_branch_%s.png' % (epoch, iteration, branch)))


def do_test_txt(cfg, model, dataset_name, step_iter, epoch, model_id):
    results = OrderedDict()
    if cfg.INPUT.VAL_RESIZE_SIZE is not None:
        mapper = DatasetMapper(cfg, is_train=False, augmentations=[T.Resize(cfg.INPUT.VAL_RESIZE_SIZE)])
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    else:
        data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = get_evaluator(cfg, dataset_name, None)
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


def inference_on_imlist(cfg, model, weights, dataset_name, prior=None):
    # Following the same detectron2.evaluation.inference_on_dataset function
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(weights)
    if cfg.INPUT.VAL_RESIZE_SIZE is not None:
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
            for output in batch_outputs:
                # Saving indexes and values of maximums instead the 20 channels scores to save memory
                output = output['sem_seg']
                output = torch.unsqueeze(output, 0)
                output = softmax2d(output)
                output = torch.squeeze(output)
                if prior is not None:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    aux = torch.from_numpy(prior).to(device)
                    output[:-1,:,:] = output[:-1,:,:]*0.75 + output[:-1,:,:]*aux[:-1,:,:]*0.25
                    conf = torch.amax(output, 0).cpu().numpy()
                else:
                    conf = torch.amax(output, 0).cpu().numpy()
                output_labels = torch.argmax(output, dim=0).to(torch.uint8).cpu().numpy()
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


def ensemble_on_imlist_and_save(cfg, modelA, modelB, weightsA, weightsB, dataset_name, img_list, save_dir,
                                evaluation=True, mask_file=None, thres=None):
    # Following the same detectron2.evaluation.inference_on_dataset function
    if cfg.INPUT.VAL_RESIZE_SIZE is not None:
        mapper = DatasetMapper(cfg, is_train=False, augmentations=[T.Resize(cfg.INPUT.VAL_RESIZE_SIZE)])
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        #modelA.apply_postprocess = False
        #modelB.apply_postprocess = False
    else:
        data_loader = build_detection_test_loader(cfg, dataset_name)
    create_folder(os.path.join(save_dir, 'predictions'))
    create_folder(os.path.join(save_dir, 'colour_predictions'))
    evaluator = get_evaluator(cfg, dataset_name, save_dir)
    evaluator.reset()
    total = len(data_loader)
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    DetectionCheckpointer(modelA, save_dir=cfg.OUTPUT_DIR).load(weightsA)
    DetectionCheckpointer(modelB, save_dir=cfg.OUTPUT_DIR).load(weightsB)
    with ExitStack() as stack:
        if isinstance(modelA, nn.Module):
            stack.enter_context(inference_context(modelA))
        if isinstance(modelB, nn.Module):
            stack.enter_context(inference_context(modelB))
        stack.enter_context(torch.no_grad())
        outputs = []
        pred_cls_num = np.zeros(num_classes)
        images_txt = os.path.join(save_dir,'image_list.txt')
        with open(images_txt,'w') as f:
            for idx, inputs in enumerate(data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0
                start_compute_time = time.perf_counter()
                outputsA = modelA(inputs)
                # Saving indexes and values of maximums instead the 20 channels scores to save memory
                outputA = outputsA[0]['sem_seg']
                outputA = torch.unsqueeze(outputA, 0)
                outputA = softmax2d(outputA)
                outputA = torch.squeeze(outputA)
                # outputA = outputA.cpu().numpy()

                outputsB = modelB(inputs)
                outputB = outputsB[0]['sem_seg']
                outputB = torch.unsqueeze(outputB, 0)
                outputB = softmax2d(outputB)
                outputB = torch.squeeze(outputB)
                # outputB = outputB.cpu().numpy()
                ensemble = torch.maximum(outputA, outputB)
                if evaluation:
                    evaluator.process(inputs, torch.unsqueeze(ensemble.argmax(dim=0).cpu(), 0), ensemble=True)
                amax_output = np.asarray(np.argmax(ensemble.cpu().numpy(), axis=0), dtype=np.uint8)
                conf = np.amax(ensemble.cpu().numpy(), axis=0)
                if thres is not None:
                    for i in range(num_classes):
                        amax_output[(conf <= thres[i])*(amax_output == i)] = 19
                        conf[(conf <= thres[i])*(amax_output == i)] = np.nan
                if mask_file is not None:
                    mask = np.asarray(Image.open(mask_file).convert('L'), dtype=bool)
                    amax_output[mask] = 19
                name = img_list[idx][0].split('/')[-1].split('.')[0]
                ensemble = Image.fromarray(amax_output)
                if cfg.INPUT.VAL_RESIZE_SIZE is not None:
                    ensemble.resize((cfg.INPUT.VAL_RESIZE_SIZE[1], cfg.INPUT.VAL_RESIZE_SIZE[0]))
                ensemble.save(os.path.join(save_dir,'predictions',name + '.png'))
                colour_label(amax_output, os.path.join(save_dir,'colour_predictions',name + '_colour.png'))
                f.write(os.path.join(save_dir,'predictions',name + '.png') + '\n')
                # np.save(os.path.join(save_dir,name + '.npy'), conf)
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
    if evaluation:
        results, conf_mat = evaluator.evaluate()
        if results is None:
            results = {}
        print_txt_format(results, '-', '-', save_dir, 'Final_ensemble')


def do_train(cfg, input_cfg, augmentation_cfg, pseudo_cfg, model, weights, train_dataset_name, test_dataset_name,
             model_id, save_checkpoints_path, epoch, cls_thresh=None, resume=False, dataset_pseudolabels=None):
    model.train()
    model.apply_postprocess = True
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
        if input_cfg.MIXED_DATA:
            mapper = DatasetMapper(cfg, is_train=True,
                                   augmentations=build_sem_seg_train_aug(input_cfg, augmentation_cfg,
                                                                         cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE),
                                   dataset2_name=input_cfg.DATASET_NAME,
                                   augmentations2=build_sem_seg_train_aug2(input_cfg, augmentation_cfg,
                                                                           cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE))
        else:
            mapper = DatasetMapper(cfg, is_train=True,
                                   augmentations=build_sem_seg_train_aug(input_cfg, augmentation_cfg,
                                                                         cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE))
    else:
        mapper = None

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        dataset: List[Dict] = DatasetCatalog.get(train_dataset_name)
        dataset_pseudo: List[Dict] = DatasetCatalog.get(dataset_pseudolabels)
        if input_cfg.RCS.ENABLED:
            sampler = RandomClassSubsampling(dataset, cfg, cfg.SOLVER.IMS_PER_BATCH)
            #sampler2 = RandomClassSubsampling(dataset2, cfg, cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1])
        else:
            sampler = TrainingSampler(len(dataset))
            #sampler2 = TrainingSampler(len(dataset2))
        mapper_pseudo = DatasetMapper(cfg, is_train=True,
                                      augmentations=build_sem_seg_train_aug(pseudo_cfg, augmentation_cfg,
                                                                            cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE))
        if cfg.SOLVER.ALTERNATE_SOURCE_PSEUDOLABELS and dataset_pseudolabels is not None \
                and cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1] > 0 \
                and cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0] > 0:
            data_loader_pseudo = build_detection_train_loader(cfg, dataset=dataset_pseudo, mapper=mapper_pseudo,
                                                              total_batch_size=cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1])
            if cfg.SOLVER.ACTIVATE_CLASSMIX:
                # If classmix is active we add "num pseudolables per batch" more samples to the source batch to use them for classmix on the pseudolabels
                data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                           total_batch_size=cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0]
                                                                            +cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1])
                training_loop_classmixdatasets(cfg, model, start_iter, max_iter, data_loader, data_loader_pseudo,
                                               storage, optimizer, scheduler, periodic_checkpointer, writers,
                                               test_dataset_name, epoch, model_id, cls_thresh)
            else:
                data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                           total_batch_size=cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0])
                training_loop_mixdatasets(cfg, model, start_iter, max_iter, data_loader, data_loader_pseudo, storage,
                                          optimizer, scheduler, periodic_checkpointer, writers, test_dataset_name,
                                          epoch, model_id)
        else:
            if cfg.SOLVER.ALTERNATE_SOURCE_PSEUDOLABELS and dataset_pseudolabels is not None \
                    and cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0] == 0:

                if cfg.SOLVER.ACTIVATE_CLASSMIX:
                    data_loader_pseudo = build_detection_train_loader(cfg, dataset=dataset_pseudo, mapper=mapper_pseudo,
                                                                      total_batch_size=cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1])
                    data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                               total_batch_size=cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0]+cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1])
                    training_loop_classmixdatasets(cfg, model, start_iter, max_iter, data_loader, data_loader_pseudo,
                                                   storage, optimizer, scheduler, periodic_checkpointer, writers,
                                                   test_dataset_name, epoch, model_id, cls_thresh)
                else:
                    data_loader = build_detection_train_loader(cfg, dataset=dataset_pseudo, mapper=mapper_pseudo, sampler=sampler,
                                                               total_batch_size=cfg.SOLVER.IMS_PER_BATCH)
                    training_loop(cfg, model, start_iter, max_iter, data_loader, storage, optimizer, scheduler,
                                  periodic_checkpointer, writers, test_dataset_name, epoch, model_id)
            else:
                data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                           total_batch_size=cfg.SOLVER.IMS_PER_BATCH)
                training_loop(cfg, model, start_iter, max_iter, data_loader, storage, optimizer, scheduler,
                              periodic_checkpointer, writers, test_dataset_name, epoch, model_id)


def training_loop(cfg, model, start_iter, max_iter, data_loader, storage, optimizer, scheduler, periodic_checkpointer, writers, test_dataset_name, epoch, model_id):
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
        if (
            cfg.TEST.EVAL_PERIOD > 0
            and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
        ):
            results = do_test_txt(cfg, model, test_dataset_name, iteration+1, epoch, model_id)
            results_list.append([results['sem_seg']['mIoU'],iteration])
            # Compared to "train_net.py", the test results are not dumped to EventStorage
            comm.synchronize()

        if iteration - start_iter > 5 and (
            (iteration + 1) % 20 == 0 or iteration == max_iter - 1
        ):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)


def training_loop_mixdatasets(cfg, model, start_iter, max_iter, data_loader, data_loader_pseudo, storage, optimizer, scheduler, periodic_checkpointer, writers, test_dataset_name, epoch, model_id):
    ''' Training loop that mixes two dataloaders to compose the final batch with the proportion specified'''
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
        if (
            cfg.TEST.EVAL_PERIOD > 0
            and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
        ):
            results = do_test_txt(cfg, model, test_dataset_name, iteration+1, epoch, model_id)
            results_list.append([results['sem_seg']['mIoU'],iteration])
            # Compared to "train_net.py", the test results are not dumped to EventStorage
            comm.synchronize()
        if iteration - start_iter > 5 and (
            (iteration + 1) % 20 == 0 or iteration == max_iter - 1
        ):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)


def compute_classmix(source, target, cls_thresh=None):
    assert len(source) == len(target)
    for idx, src_item in enumerate(source):
        classes = torch.unique(src_item['sem_seg'])
        if cls_thresh is not None:
            cls_aux=[]
            cls_per_confidence = np.argsort(cls_thresh)
            for i in range(len(cls_per_confidence)):
                if cls_per_confidence[i] in classes and cls_thresh[cls_per_confidence[i]] != 0:
                    cls_aux.append(cls_per_confidence[i])
                    if len(cls_aux) == int(len(classes)/2):
                        break
            cls_sel=[]
            for cls in classes:
                if cls not in cls_aux:
                    cls_sel.append(cls)
            cls_sel = torch.from_numpy(np.asarray(cls_sel))
        else:
            cls_sel=classes[torch.randperm(len(classes))][:int(len(classes)/2)]
        idx_classes = (src_item['sem_seg'][..., None] != cls_sel).all(-1)
        target[idx]['sem_seg'][idx_classes] = src_item['sem_seg'][idx_classes]
        target[idx]['image'][:,idx_classes] = src_item['image'][:,idx_classes]
    return target


def training_loop_classmixdatasets(cfg, model, start_iter, max_iter, data_loader, data_loader_pseudo, storage, optimizer, scheduler, periodic_checkpointer, writers, test_dataset_name, epoch, model_id, cls_thresh):
    ''' Training loop that mixes two dataloaders to compose the final batch with the proportion specified'''
    results_list = []
    for data1, data2, iteration in zip(data_loader, data_loader_pseudo, range(start_iter, max_iter)):
        pseudo_bz=len(data2)
        if cfg.SOLVER.CLASSMIX_ON_CLASS_DEMAND:
            data2 = compute_classmix(data1[-pseudo_bz:], data2, cls_thresh)
        else:
            data2 = compute_classmix(data1[-pseudo_bz:], data2)
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
        if (
            cfg.TEST.EVAL_PERIOD > 0
            and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
        ):
            results = do_test_txt(cfg, model, test_dataset_name, iteration+1, epoch, model_id)
            results_list.append([results['sem_seg']['mIoU'],iteration])
            # Compared to "train_net.py", the test results are not dumped to EventStorage
            comm.synchronize()

        if iteration - start_iter > 5 and (
            (iteration + 1) % 20 == 0 or iteration == max_iter - 1
        ):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.unlabeled_dataset_A is not None:
        cfg.DATASETS.UNLABELED_DATASET_A = args.unlabeled_dataset_A
    if args.unlabeled_dataset_B is not None:
        cfg.DATASETS.UNLABELED_DATASET_B = args.unlabeled_dataset_B
    if args.weights_branchA is not None:
        cfg.MODEL.WEIGHTS_BRANCH_A = args.weights_branchA
    if args.weights_branchB is not None:
        cfg.MODEL.WEIGHTS_BRANCH_B = args.weights_branchB
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
    for i in range(num_classes):
        x = pred_conf[pred_cls_num == i]
        if len(x) == 0:
            threshold.append(0)
            continue
        x = np.sort(x)
        if type(tgt_portion) == np.ndarray:
            threshold.append(x[np.int(np.round(len(x)*(1-tgt_portion[i])))])
        else:
            threshold.append(x[np.int(np.round(len(x)*(1-tgt_portion)))])
    threshold = np.array(threshold)
    threshold[threshold > 0.95] = 0.95
    threshold[threshold < 0.5] = 0.5
    return threshold


def apply_mpt(outputs, num_classes, tgt_num, tgt_portion, mask=None, prior=None, prior_thres=0):
    pred_cls_num = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.uint8)
    pred_conf = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.float32)
    for index, output in enumerate(outputs):
        pred_cls_num[index] = output[0]
        pred_conf[index] = output[1]
    thres = compute_mtp_thresholds(pred_conf, pred_cls_num, tgt_portion, num_classes)
    logger.info("MPT tgt_portion: {}".format(tgt_portion))
    logger.info("MPT thresholds: {}".format(thres))
    logger.info("MPT prior threshold: {}".format(prior_thres))
    pseudolabels = []
    pseudolabels_not_filtered = []
    scores_list = []
    for index in range(tgt_num):
        # create aux array for mean class confidence (0 to num_classes -1), pixel per class (num_classes to num_classes*2-1) count and scores (num_classes*2)
        aux_scores = np.zeros((num_classes*2+1), dtype=np.float32)
        pseudolabels_not_filtered.append(pred_cls_num[index])
        label = pred_cls_num[index].copy()
        # Apply mask to the pseudolabel (useful to remove detection on prefixed void parts (e.g. ego vehicle))
        if mask is not None:
            label[mask] = 19
        prob = pred_conf[index]
        for i in range(num_classes):
            if prior is not None and prior_thres > 0:
                prior_conf_mask = prior[i,:,:].copy()
                prior_conf_mask[prior[i,:,:] >= prior_thres] = 1.0
                prior_conf_mask[prior[i,:,:] < prior_thres] *= 1.0/prior_thres
                # aux = prob*0.85 + prob*prior[i,:,:]*0.15
                aux = prob*prior_conf_mask
                label[(aux<=thres[i])*(label==i)] = 19 # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(aux<=thres[i])*(label==i)] = np.nan
            else:
                label[(prob<=thres[i])*(label==i)] = 19 # '255' in cityscapes indicates 'unlabaled' for trainIDs
                prob[(prob<=thres[i])*(label==i)] = np.nan
            class_conf = prob*(label==i)
            class_conf[class_conf == 0] = np.nan
            aux_scores[i] = np.nanmean(class_conf)
        pseudolabels.append(label)
        #Compute image score using the mean of the weighted confidence pixels values higher than the threshold cls_thresh
        classes_id, pixel_count = np.unique(label, return_counts=True)
        score = np.nanmean(prob)
        aux_scores[-1] = score
        for idx, class_id in enumerate(classes_id):
            aux_scores[class_id+num_classes] = pixel_count[idx]
        scores_list.append(aux_scores)
    return pseudolabels, scores_list, pseudolabels_not_filtered, thres


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


def colour_label(inference, filename):
    pred_colour = 255 * np.ones([inference.shape[0],inference.shape[1],3], dtype=np.uint8)
    for train_id, label in trainId2label.items():
        pred_colour[(inference == train_id),0] = label.color[0]
        pred_colour[(inference == train_id),1] = label.color[1]
        pred_colour[(inference == train_id),2] = label.color[2]
    color_map = []
    classes = []
    for i in range(len(trainId2label)):
        if i in trainId2label:
            color_map.append(trainId2label[i].color)
            classes.append(trainId2label[i].name)
    legend = draw_legend(inference.shape[1], color_map, classes, n_lines=2)
    pred_colour = np.concatenate((pred_colour, legend))
    Image.fromarray(pred_colour).save(filename)


def save_pseudolabels(images, pseudolabels, scores, pseudolabels_path, coloured_pseudolabels_path=None,
                      pseudolabels_not_filtered=None, coloured_pseudolabels_not_filtered_path=None, file_text=''):
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
                    filename = image[0].split('/')[-1].split('.')[0] + '.png'
                    Image.fromarray(pseudolabels[idx]).save(os.path.join(pseudolabels_path,filename))
                    if coloured_pseudolabels_path is not None:
                        colour_label(pseudolabels[idx], os.path.join(coloured_pseudolabels_path,filename))
                    if pseudolabels_not_filtered is not None and coloured_pseudolabels_not_filtered_path is not None:
                        colour_label(pseudolabels_not_filtered[idx],
                                     os.path.join(coloured_pseudolabels_not_filtered_path, filename))
                    # Create txt with files names and scores
                    f.write('%s %s %s %s\n' % (filename, str(scores[idx][-1]), str(scores[idx][-2]),
                                               str(np.count_nonzero(scores[idx][:19]))))
                    g.write('%s\n' % (image[0]))
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
                                    new_images_txt, new_labels_txt, save_img_txt, save_labels_txt, save_scores_txt, sorting_method):
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
        new_scores_dataset.write(accum_scores[idx][0] + ' ' + accum_scores[idx][1] + ' ' + accum_scores[idx][2] + ' ' + accum_scores[idx][3] + '\n')
    new_img_dataset.close()
    new_labels_dataset.close()
    new_scores_dataset.close()
    return save_img_txt, save_labels_txt, save_scores_txt


def compute_by_confidence_on_class_demand(cls_threshA, info_inference_B, scoresB, max_selected=0.20):
    idxsort_thres = np.argsort(cls_threshA)  # exclude class void
    sorted_idx = []
    for idx in idxsort_thres:
        # Select from branch B images with less confidence class from branch A
        img_id_list = np.asarray(info_inference_B["summary"][idx])
        if len(img_id_list) == 0:
            continue
        # Order image selection by confidence higher to lower
        idx_per_confidence = np.argsort(scoresB[img_id_list][:, idx])[::-1]
        # Compute difference between best and worst image
        diff_confidence = scoresB[idx_per_confidence][:, idx][0] - scoresB[idx_per_confidence][:, idx][-1]
        # Compute threshold to ensure confidences close to the top image
        confidence_thres = scoresB[idx_per_confidence][:, idx][0] - (diff_confidence*max_selected)
        # Obtain idx to the images ordered by confidence
        final_sort = idx_per_confidence[scoresB[idx_per_confidence][:, idx] > confidence_thres]
        sorted_idx.extend(final_sort)
    _, idx = np.unique(sorted_idx, return_index=True)
    return np.asarray(sorted_idx)[np.sort(idx)]


def sorting_scores(scoresA, scoresB, sorting_method, num_classes, info_inference_A, info_inference_B, cls_threshA, cls_threshB, idx_to_remove=None, selftraining=False):
    if sorting_method == 'per_class':
        sorted_idxA = np.lexsort((scoresA[:,-1],np.count_nonzero(scoresA[:,num_classes:-2], axis=1)))[::-1]
        sorted_idxB = np.lexsort((scoresB[:,-1],np.count_nonzero(scoresB[:,num_classes:-2], axis=1)))[::-1]
    elif sorting_method == 'per_void_pixels':
        # Sorting by number of void pixels (lower to higher)
        sorted_idxA = np.argsort(scoresA[:,-2])
        sorted_idxB = np.argsort(scoresB[:,-2])
    elif sorting_method == 'cotraining_confidence_score':
        # Sorting by score determined by the confidence difference of each class between branches
        conf_A = scoresA[:,0:num_classes]
        conf_B = scoresB[:,0:num_classes]
        conf_diffA = conf_B - conf_A
        conf_diffA[(conf_diffA < 0)] = 0
        conf_diffB = conf_A - conf_B
        conf_diffB[(conf_diffB < 0)] = 0
        new_scoresA = np.nanmean(conf_diffA * (1 - conf_A), axis=1)
        new_scoresB = np.nanmean(conf_diffB * (1 - conf_B), axis=1)
        sorted_idxA = np.lexsort((np.count_nonzero(scoresA[:,num_classes:-2], axis=1), new_scoresA))[::-1]
        sorted_idxB = np.lexsort((np.count_nonzero(scoresB[:,num_classes:-2], axis=1), new_scoresB))[::-1]
    elif sorting_method == 'by_confidence_on_class_demand':
        #Order by less confident classes
        sorted_idxA = compute_by_confidence_on_class_demand(cls_threshB, info_inference_A, scoresA)
        sorted_idxB = compute_by_confidence_on_class_demand(cls_threshA, info_inference_B, scoresB)
    elif sorting_method == 'by_confidence_difference_between_branches':
        #Order by less confident classes
        sorted_idxA = compute_by_confidence_on_class_demand(cls_threshB - cls_threshA, info_inference_A, scoresA)
        sorted_idxB = compute_by_confidence_on_class_demand(cls_threshA - cls_threshB, info_inference_B, scoresB)
    elif sorting_method == 'confidence':
        # Sorting by confidence lower to higher
        sorted_idxA = np.argsort(scoresA[:,-1])
        sorted_idxB = np.argsort(scoresB[:,-1])
        # higher to lower
        sorted_idxA = sorted_idxA[::-1][:len(scoresA)]
        sorted_idxB = sorted_idxB[::-1][:len(scoresB)]
    else:
        #No sorting
        sorted_idxA = np.arange(len(scoresA))
        sorted_idxB = np.arange(len(scoresB))
    # Delete idx not desired from filtering
    if idx_to_remove is not None and len(idx_to_remove) > 0:
        idx = np.where(np.in1d(sorted_idxA, idx_to_remove))[0]
        sorted_idxA = np.concatenate((np.delete(sorted_idxA, idx), sorted_idxA[idx]), axis=0)
        idx = np.where(np.in1d(sorted_idxB, idx_to_remove))[0]
        sorted_idxB = np.concatenate((np.delete(sorted_idxB, idx), sorted_idxB[idx]), axis=0)
    return sorted_idxA, sorted_idxB


def get_data(data_list):
    with open(data_list,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    return im_list


def compute_statistics(labels, dataset=None, inference_mode=False, min_pixels=199):
    images = []
    categories = {}
    summary = {}
    info = {"images":images, "categories":categories, "summary":summary}
    if not inference_mode:
        dataset = get_data(labels)
    for train_id, label in trainId2label.items():
        if train_id >= 0:
            categories[train_id] = label[0]
            summary[train_id] = []
    summary[19] = []
    categories[19] = 'Void'
    for i in range(len(dataset)):
        if inference_mode:
            inference = labels[i]
        else:
            inference = np.asarray(Image.open(dataset[i][0]).convert('L'), dtype=np.uint8)
        classes = np.unique(inference, return_counts=True)
        image_dict = {  "id": i,
                    "file_name": dataset[i][0],
                    "classes": classes[0].tolist(),
                    "pixels": classes[1].tolist(),
                    "pixels_perc": (classes[1]/(inference.shape[1]*inference.shape[0])).tolist(),
                    "width": inference.shape[1],
                    "height": inference.shape[0]}
        images.append(image_dict)
        for idx, obj_cls in enumerate(classes[0]):
            if classes[1][idx] > min_pixels:
                summary[obj_cls].append(i)
        #print('%d/%d' % (i+1,len(labels)))
    logger.info("\n --- Summary --- \n")
    for idx, obj_cls in enumerate(categories):
        logger.info("Class %s: %d images, %.2f%%" % (categories[obj_cls], len(summary[obj_cls]), len(summary[obj_cls])*100/len(dataset)))
    logger.info("Total images: %d" % (len(dataset)))
    return info


def compute_ensembles(cfg, pseudolabels_A, pseudolabels_B, mode='+'):
    logger.info("Computing ensembling of pseudolabels")
    start_time = time.perf_counter()
    for idx, _ in enumerate(pseudolabels_A):
        aux_pseudoA = pseudolabels_A[idx].copy()
        aux_pseudoB = pseudolabels_B[idx].copy()
        # Common labels indexes
        idx_common = pseudolabels_A[idx] == pseudolabels_B[idx]
        # Not shared labels indexes
        idx_noncommon = pseudolabels_A[idx] != pseudolabels_B[idx]
        # From not shared labels indexes where A have a valid label and B is void
        idxA_novoid = (pseudolabels_B[idx] == cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE) * idx_noncommon
        # From not shared labels indexes where B have a valid label and A is void
        idxB_novoid = (pseudolabels_A[idx] == cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE) * idx_noncommon
        # Not shared labels indexes without void labels
        idx_noncommon = idx_noncommon * np.logical_not(np.logical_or(idxA_novoid, idxB_novoid))
        # Indexes from A common with B and indexes where A is void and B have a value
        idxA = np.logical_or(idx_common, idxB_novoid)
        # Indexes from B common with A and indexes where B is void and A have a value
        idxB = np.logical_or(idx_common, idxA_novoid)
        # Assign to aux A values from B where labels are common and A is void
        aux_pseudoA[idxA] = pseudolabels_B[idx][idxA]
        aux_pseudoB[idxB] = pseudolabels_A[idx][idxB]
        if mode == '+':
            # Assign to aux A values from A that differs with B
            aux_pseudoA[idx_noncommon] = pseudolabels_A[idx][idx_noncommon]
            aux_pseudoB[idx_noncommon] = pseudolabels_B[idx][idx_noncommon]
        else:
            # Assign not shared labels as void
            aux_pseudoA[idx_noncommon] = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
            aux_pseudoB[idx_noncommon] = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        pseudolabels_A[idx] = aux_pseudoA
        pseudolabels_B[idx] = aux_pseudoB
    total_time = time.perf_counter() - start_time
    logger.info("Ensembles done in {:.2f} s".format(total_time))
    return pseudolabels_A, pseudolabels_B


def select_unlabeled_data(cfg, args, seed, epoch):
    seed = seed + epoch
    # prepare unlabeled data
    logger.info("Seed for unlabeled data {}".format(seed))
    # Read unlabeled data from the txt specified and select randomly X samples defined on max_unlabeled_samples
    unlabeled_datasetA = get_unlabeled_data(cfg.DATASETS.UNLABELED_DATASET_A, args.step_inc, seed, cfg.DATASETS.MAX_UNLABELED_SAMPLES)
    logger.info("Unlabeled data selected from {}: {}".format(cfg.DATASETS.UNLABELED_DATASET_A,len(unlabeled_datasetA)))
    if args.same_domain:
        unlabeled_datasetB = unlabeled_datasetA
    else:
        unlabeled_datasetB = get_unlabeled_data(cfg.DATASETS.UNLABELED_DATASET_B, args.step_inc, seed, cfg.DATASETS.MAX_UNLABELED_SAMPLES)
        logger.info("Unlabeled data selected from {}: {}".format(cfg.DATASETS.UNLABELED_DATASET_B,len(unlabeled_datasetB)))
    logger.info("Unlabeled data selected from A: {}".format(len(unlabeled_datasetA)))
    logger.info("Unlabeled data selected from B: {}".format(len(unlabeled_datasetB)))
    # Regiter unlabeled dataset on detectron 2
    built_inference_dataset(cfg, unlabeled_datasetA, args.unlabeled_dataset_A_name)
    built_inference_dataset(cfg, unlabeled_datasetB, args.unlabeled_dataset_B_name)
    return unlabeled_datasetA, unlabeled_datasetB


def generate_pseudolabels(cfg, model, weights_inference_branch, unlabeled_dataset_name, unlabeled_dataset,
                          tgt_portion, source_priors, prior_thres):
    start_time = time.perf_counter()
    # Inference return a tuple of labels and confidences
    inference = inference_on_imlist(cfg, model, weights_inference_branch, unlabeled_dataset_name)
    total_time = time.perf_counter() - start_time
    logger.info("Compute inference on unlabeled dataset: {:.2f} s".format(total_time))
    logger.info("Pseudolabeling mode: {}, Threshold: {}".format(cfg.PSEUDOLABELING.MODE, tgt_portion))
    if args.mask_file is not None:
        mask = Image.open(args.mask_file).convert('L')
        mask = mask.resize((cfg.INPUT_PSEUDO.RESIZE_SIZE[1], cfg.INPUT_PSEUDO.RESIZE_SIZE[0]))
        mask = np.asarray(mask, dtype=bool)
    else:
        mask = None
    if cfg.PSEUDOLABELING.MODE == 'mpt':
        start_time = time.perf_counter()
        pseudolabels, scores_list, pseudolabels_not_filtered, cls_thresh = apply_mpt(inference,
                                                cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, len(unlabeled_dataset),
                                                tgt_portion, mask, source_priors, prior_thres)
        total_time = time.perf_counter() - start_time
        logger.info("MPT on unlabeled dataset: {:.2f} s".format(total_time))
    else:
        raise Exception('unknown pseudolabeling method defined')

    return pseudolabels, scores_list, pseudolabels_not_filtered, cls_thresh


def recompute_pseudolabels(cfg, model_name, model, weights_inference_branch, epoch, accumulated_selection_img, accumulated_selection_pseudo,
                           dataset_target, tgt_portion, source_priors, prior_thres, accumulated_scores, dataset_path):
    logger.info("Recompute accumulated pseudolabels and update")
    re_pseudolabels_path_model = os.path.join(cfg.OUTPUT_DIR, model_name, str(epoch), 'pseudolabeling/recomputed_pseudolabels')
    create_folder(re_pseudolabels_path_model)
    re_coloured_pseudolabels_path = os.path.join(cfg.OUTPUT_DIR, model_name, str(epoch), 'pseudolabeling/recomputed_coloured_pseudolabels')
    create_folder(re_coloured_pseudolabels_path)
    accumulated_img = get_data(accumulated_selection_img)
    accumulated_pseudo = get_data(accumulated_selection_pseudo)
    pseudolabels, scores_list, _, _, = generate_pseudolabels(cfg, model, weights_inference_branch,
                                             dataset_target, accumulated_pseudo, tgt_portion, source_priors, prior_thres)
    images_txt, psedolabels_txt, filenames_and_scores = save_pseudolabels(accumulated_img,
                    pseudolabels, scores_list, re_pseudolabels_path_model, re_coloured_pseudolabels_path, file_text='recomp_')
    _, _, _ = update_best_score_txts_and_save(accumulated_scores, accumulated_selection_img, accumulated_selection_pseudo,
                                    filenames_and_scores, images_txt, psedolabels_txt,
                                    os.path.join(dataset_path,'dataset_img.txt'),
                                    os.path.join(dataset_path,'dataset_pseudolabels.txt'),
                                    os.path.join(dataset_path,'filenames_and_scores.txt'), cfg.PSEUDOLABELING.SORTING)


def main(args):
    cfg = setup(args)
    continue_epoch = args.continue_epoch
    accumulated_selection_imgA = []
    accumulated_selection_pseudoA = []
    accumulated_selection_imgB = []
    accumulated_selection_pseudoB = []
    collaboration = cfg.PSEUDOLABELING.COLLABORATION
    accumulation_mode = cfg.PSEUDOLABELING.ACCUMULATION
    num_selected = cfg.PSEUDOLABELING.NUMBER
    weights_inference_branchA = cfg.MODEL.WEIGHTS_BRANCH_A
    weights_train_branchA = cfg.MODEL.WEIGHTS_BRANCH_A
    weights_inference_branchB = cfg.MODEL.WEIGHTS_BRANCH_B
    weights_train_branchB = cfg.MODEL.WEIGHTS_BRANCH_B
    tgt_portion = cfg.PSEUDOLABELING.INIT_TGT_PORT
    if type(tgt_portion) == list:
        tgt_portion = np.asarray(tgt_portion)
        max_list_tgt = tgt_portion + cfg.PSEUDOLABELING.MAX_TGT_PORT
    if cfg.PSEUDOLABELING.INIT_TGT_PORT_B is not None:
        tgt_portion_B = cfg.PSEUDOLABELING.INIT_TGT_PORT_B
        if type(tgt_portion_B) == list:
            tgt_portion_B = np.asarray(cfg.PSEUDOLABELING.INIT_TGT_PORT_B)
            tgt_portion_B = np.asarray(tgt_portion_B)
            max_list_tgt_B = tgt_portion_B + cfg.PSEUDOLABELING.MAX_TGT_PORT_B
    else:
        tgt_portion_B = tgt_portion
        if type(tgt_portion) == list:
            max_list_tgt_B = max_list_tgt
    # Set initial scores to surpass during an epoch to propagate weghts to the next one
    source_img_datasetA = cfg.DATASETS.TRAIN_IMG_TXT
    source_gt_datasetA = cfg.DATASETS.TRAIN_GT_TXT
    if cfg.DATASETS.TRAIN_IMG_TXT2 is not None:
        source_img_datasetB = cfg.DATASETS.TRAIN_IMG_TXT2
    else:
        source_img_datasetB = cfg.DATASETS.TRAIN_IMG_TXT
    if cfg.DATASETS.TRAIN_GT_TXT2 is not None:
        source_gt_datasetB = cfg.DATASETS.TRAIN_GT_TXT2
    else:
        source_gt_datasetB = cfg.DATASETS.TRAIN_GT_TXT
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

    # Start co-training
    if not args.no_training:
        for epoch in range(args.continue_epoch,args.epochs):
            if continue_epoch > 0:
                weights_inference_branchA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch-1),'checkpoints/model_final.pth')
                weights_inference_branchB = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch-1),'checkpoints/model_final.pth')
                if type(tgt_portion) == np.ndarray:
                    tgt_portion = np.where(tgt_portion >= max_list_tgt,
                                           max_list_tgt,
                                           tgt_portion + cfg.PSEUDOLABELING.TGT_PORT_STEP*continue_epoch)
                else:
                    tgt_portion = min(tgt_portion + cfg.PSEUDOLABELING.TGT_PORT_STEP*continue_epoch,
                                      cfg.PSEUDOLABELING.MAX_TGT_PORT)
                if cfg.PSEUDOLABELING.INIT_TGT_PORT_B is not None:
                    if type(tgt_portion_B) == np.ndarray:
                        tgt_portion_B = np.where(tgt_portion_B >= max_list_tgt_B,
                                           max_list_tgt_B,
                                           tgt_portion_B + cfg.PSEUDOLABELING.TGT_PORT_STEP_B*continue_epoch)
                    else:
                        tgt_portion_B = min(tgt_portion_B + cfg.PSEUDOLABELING.TGT_PORT_STEP_B*continue_epoch,
                                          cfg.PSEUDOLABELING.MAX_TGT_PORT_B)
                else:
                    tgt_portion_B = tgt_portion
                prior_thres = max(prior_thres-(prior_relax*continue_epoch), 0)
            logger.info("Starting training from iteration {}".format(epoch))
            logger.info("prepare unlabeled data")
            unlabeled_datasetA, unlabeled_datasetB = select_unlabeled_data(cfg, args, seed, epoch)
            # Compute inference on unlabeled datasets
            model = build_model(cfg)
            logger.info("Compute inference on unlabeled data with branch A")
            pseudolabels_A, scores_listA, pseudolabels_A_not_filtered, cls_threshA = \
                    generate_pseudolabels(cfg, model, weights_inference_branchA,
                                          args.unlabeled_dataset_A_name, unlabeled_datasetA,
                                          tgt_portion, source_priors, prior_thres)
            logger.info("Compute inference on unlabeled data with branch B")
            pseudolabels_B, scores_listB, pseudolabels_B_not_filtered, cls_threshB = \
                    generate_pseudolabels(cfg, model, weights_inference_branchB,
                                          args.unlabeled_dataset_B_name, unlabeled_datasetB,
                                          tgt_portion_B, source_priors, prior_thres)

            if args.ensembles and epoch > 0:
                if args.ensembles_subtraction:
                    pseudolabels_A, pseudolabels_B = compute_ensembles(cfg, pseudolabels_A, pseudolabels_B, '-')
                else:
                    pseudolabels_A, pseudolabels_B = compute_ensembles(cfg, pseudolabels_A, pseudolabels_B)

            logger.info("Computing pseudolabels statistics")
            start_time = time.perf_counter()
            info_inference_A = compute_statistics(pseudolabels_A, unlabeled_datasetA, True, min_pixels=args.min_pixels)
            total_time = time.perf_counter() - start_time
            logger.info("Statistics from branch A in {:.2f} s".format(total_time))
            start_time = time.perf_counter()
            info_inference_B = compute_statistics(pseudolabels_B, unlabeled_datasetB, True, min_pixels=args.min_pixels)
            total_time = time.perf_counter() - start_time
            logger.info("Statistics from branch B in {:.2f} s".format(total_time))

            #path pseudolabels
            pseudolabels_path_model_A = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'pseudolabeling/pseudolabels')
            create_folder(pseudolabels_path_model_A)
            coloured_pseudolabels_path_model_A = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'pseudolabeling/coloured_pseudolabels')
            create_folder(coloured_pseudolabels_path_model_A)
            coloured_pseudolabels_not_filtered_path_model_A = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'pseudolabeling/coloured_pseudolabels_not_filtered')
            create_folder(coloured_pseudolabels_not_filtered_path_model_A)
            pseudolabels_path_model_B = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch),'pseudolabeling/pseudolabels')
            create_folder(pseudolabels_path_model_B)
            coloured_pseudolabels_path_model_B = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch),'pseudolabeling/coloured_pseudolabels')
            create_folder(coloured_pseudolabels_path_model_B)
            coloured_pseudolabels_not_filtered_path_model_B = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch),'pseudolabeling/coloured_pseudolabels_not_filtered')
            create_folder(coloured_pseudolabels_not_filtered_path_model_B)
            dataset_A_path = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'unlabeled_data_selected')
            create_folder(dataset_A_path)
            dataset_B_path = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch),'unlabeled_data_selected')
            create_folder(dataset_B_path)
            checkpoints_A_path = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'checkpoints')
            create_folder(checkpoints_A_path)
            checkpoints_B_path = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch),'checkpoints')
            create_folder(checkpoints_B_path)

            # Continue cotraining on the specified epoch
            if continue_epoch > 0:
                accumulated_selection_imgA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch-1),'unlabeled_data_selected/dataset_img.txt')
                accumulated_selection_pseudoA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch-1),'unlabeled_data_selected/dataset_pseudolabels.txt')
                accumulated_scores_A = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch-1),'unlabeled_data_selected/filenames_and_scores.txt')
                accumulated_selection_imgB = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch-1),'unlabeled_data_selected/dataset_img.txt')
                accumulated_selection_pseudoB = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch-1),'unlabeled_data_selected/dataset_pseudolabels.txt')
                accumulated_scores_B = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch-1),'unlabeled_data_selected/filenames_and_scores.txt')
                continue_epoch = 0

            logger.info("Collaboration mode: {}".format(collaboration))
            scores_listA = np.asarray(scores_listA)
            pseudolabels_A = np.asarray(pseudolabels_A)
            unlabeled_datasetA = np.asarray(unlabeled_datasetA)
            pseudolabels_A_not_filtered = np.asarray(pseudolabels_A_not_filtered)
            scores_listB = np.asarray(scores_listB)
            pseudolabels_B = np.asarray(pseudolabels_B)
            unlabeled_datasetB = np.asarray(unlabeled_datasetB)
            pseudolabels_B_not_filtered = np.asarray(pseudolabels_B_not_filtered)

            if len(unlabeled_datasetA) < cfg.PSEUDOLABELING.NUMBER:
                num_selected = len(unlabeled_datasetA)

            logger.info("Sorting mode: {}".format(cfg.PSEUDOLABELING.SORTING))
            start_time = time.perf_counter()
            if "self" in collaboration.lower(): #Self-training for each branch
                # Order pseudolabels by confidences (scores) higher to lower and select number defined to merge with source data
                sorted_idxA, sorted_idxB = sorting_scores(scores_listA, scores_listB, cfg.PSEUDOLABELING.SORTING, cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, info_inference_A, info_inference_B, cls_threshA, cls_threshB, selftraining=True)

                sorted_scores_listA = scores_listA[sorted_idxA][:num_selected]
                sorted_pseudolabels_A = pseudolabels_A[sorted_idxA][:num_selected]
                sorted_unlabeled_datasetA = unlabeled_datasetA[sorted_idxA][:num_selected]
                sorted_pseudolabels_A_not_filtered = pseudolabels_A_not_filtered[sorted_idxA][:num_selected]

                sorted_scores_listB = scores_listB[sorted_idxB][:num_selected]
                sorted_pseudolabels_B = pseudolabels_B[sorted_idxB][:num_selected]
                sorted_unlabeled_datasetB = unlabeled_datasetB[sorted_idxB][:num_selected]
                sorted_pseudolabels_B_not_filtered = pseudolabels_B_not_filtered[sorted_idxB][:num_selected]

            if "cotraining" in collaboration.lower():
                # Order pseudolabels by confidence lower to higher and asign the less n confident to the other model
                sorted_idxA, sorted_idxB = sorting_scores(scores_listA, scores_listB, cfg.PSEUDOLABELING.SORTING, cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, info_inference_A, info_inference_B, cls_threshA, cls_threshB)
                if "self" in collaboration.lower():
                    sorted_scores_listA = np.concatenate((sorted_scores_listA[:int(num_selected/2)], scores_listB[sorted_idxB][:int(num_selected/2)]), axis=0)
                    sorted_pseudolabels_A = np.concatenate((sorted_pseudolabels_A[:int(num_selected/2)], pseudolabels_B[sorted_idxB][:int(num_selected/2)]), axis=0)
                    sorted_unlabeled_datasetA = np.concatenate((sorted_unlabeled_datasetA[:int(num_selected/2)], unlabeled_datasetA[sorted_idxB][:int(num_selected/2)]), axis=0)
                    sorted_pseudolabels_A_not_filtered = np.concatenate((sorted_pseudolabels_A_not_filtered[:int(num_selected/2)], pseudolabels_B_not_filtered[sorted_idxB][:int(num_selected/2)]), axis=0)

                    sorted_scores_listB = np.concatenate((sorted_scores_listB[:int(num_selected/2)], scores_listA[sorted_idxA][:int(num_selected/2)]), axis=0)
                    sorted_pseudolabels_B = np.concatenate((sorted_pseudolabels_B[:int(num_selected/2)], pseudolabels_A[sorted_idxA][:int(num_selected/2)]), axis=0)
                    sorted_unlabeled_datasetB = np.concatenate((sorted_unlabeled_datasetB[:int(num_selected/2)], unlabeled_datasetB[sorted_idxA][:int(num_selected/2)]), axis=0)
                    sorted_pseudolabels_B_not_filtered = np.concatenate((sorted_pseudolabels_B_not_filtered[:int(num_selected/2)], pseudolabels_A_not_filtered[sorted_idxA][:int(num_selected/2)]), axis=0)
                else:
                    sorted_scores_listA = scores_listB[sorted_idxB][:num_selected]
                    sorted_pseudolabels_A = pseudolabels_B[sorted_idxB][:num_selected]
                    sorted_unlabeled_datasetA = unlabeled_datasetA[sorted_idxB][:num_selected]
                    sorted_pseudolabels_A_not_filtered = pseudolabels_B_not_filtered[sorted_idxB][:num_selected]

                    sorted_scores_listB = scores_listA[sorted_idxA][:num_selected]
                    sorted_pseudolabels_B = pseudolabels_A[sorted_idxA][:num_selected]
                    sorted_unlabeled_datasetB = unlabeled_datasetB[sorted_idxA][:num_selected]
                    sorted_pseudolabels_B_not_filtered = pseudolabels_A_not_filtered[sorted_idxA][:num_selected]

            if not "self" in collaboration.lower() and not "cotraining" in collaboration.lower():
                raise Exception('unknown collaboration of models defined')

            total_time = time.perf_counter() - start_time
            logger.info("Sorting done in {:.2f} s".format(total_time))

            # free memory
            del scores_listA
            del pseudolabels_A
            del unlabeled_datasetA
            del pseudolabels_A_not_filtered
            del scores_listB
            del pseudolabels_B
            del unlabeled_datasetB
            del pseudolabels_B_not_filtered
            gc.collect()

            # select candidates and save them to add them to the source data
            images_txt_A, psedolabels_txt_A, filenames_and_scoresA = save_pseudolabels(sorted_unlabeled_datasetA, sorted_pseudolabels_A, sorted_scores_listA, pseudolabels_path_model_A,
                coloured_pseudolabels_path_model_A, sorted_pseudolabels_A_not_filtered, coloured_pseudolabels_not_filtered_path_model_A)
            images_txt_B, psedolabels_txt_B, filenames_and_scoresB = save_pseudolabels(sorted_unlabeled_datasetB, sorted_pseudolabels_B, sorted_scores_listB, pseudolabels_path_model_B,
                coloured_pseudolabels_path_model_B, sorted_pseudolabels_B_not_filtered, coloured_pseudolabels_not_filtered_path_model_B)

            # free memory
            del sorted_unlabeled_datasetA
            del sorted_pseudolabels_A
            del sorted_scores_listA
            del sorted_pseudolabels_A_not_filtered
            del sorted_unlabeled_datasetB
            del sorted_pseudolabels_B
            del sorted_scores_listB
            del sorted_pseudolabels_B_not_filtered
            gc.collect()

            # Compute data accumulation procedure
            logger.info("Acumulation mode: {}".format(accumulation_mode.lower()))
            start_time = time.perf_counter()
            if accumulation_mode is not None and len(accumulated_selection_imgA) > 0:
                if accumulation_mode.lower() == 'all':
                    accumulated_selection_imgA = merge_txts_and_save(os.path.join(dataset_A_path,'dataset_img.txt'),
                                                                        accumulated_selection_imgA, images_txt_A)
                    accumulated_selection_pseudoA = merge_txts_and_save(os.path.join(dataset_A_path,'dataset_pseudolabels.txt'),
                                                                        accumulated_selection_pseudoA, psedolabels_txt_A)
                    accumulated_scores_A = merge_txts_and_save(os.path.join(dataset_A_path,'filenames_and_scores.txt'),
                                                                        accumulated_scores_A, filenames_and_scoresA)
                    accumulated_selection_imgB = merge_txts_and_save(os.path.join(dataset_B_path,'dataset_img.txt'),
                                                                        accumulated_selection_imgB, images_txt_B)
                    accumulated_selection_pseudoB = merge_txts_and_save(os.path.join(dataset_B_path,'dataset_pseudolabels.txt'),
                                                                        accumulated_selection_pseudoB, psedolabels_txt_B)
                    accumulated_scores_B = merge_txts_and_save(os.path.join(dataset_B_path,'filenames_and_scores.txt'),
                                                                        accumulated_scores_B, filenames_and_scoresB)
                elif accumulation_mode.lower() == 'update_best_score':
                    accumulated_selection_imgA, accumulated_selection_pseudoA, accumulated_scores_A = update_best_score_txts_and_save(
                                                    accumulated_scores_A, accumulated_selection_imgA, accumulated_selection_pseudoA,
                                                    filenames_and_scoresA, images_txt_A, psedolabels_txt_A,
                                                    os.path.join(dataset_A_path,'dataset_img.txt'),
                                                    os.path.join(dataset_A_path,'dataset_pseudolabels.txt'),
                                                    os.path.join(dataset_A_path,'filenames_and_scores.txt'), cfg.PSEUDOLABELING.SORTING)
                    accumulated_selection_imgB, accumulated_selection_pseudoB, accumulated_scores_B = update_best_score_txts_and_save(
                                                    accumulated_scores_B, accumulated_selection_imgB, accumulated_selection_pseudoB,
                                                    filenames_and_scoresB, images_txt_B, psedolabels_txt_B,
                                                    os.path.join(dataset_B_path,'dataset_img.txt'),
                                                    os.path.join(dataset_B_path,'dataset_pseudolabels.txt'),
                                                    os.path.join(dataset_B_path,'filenames_and_scores.txt'), cfg.PSEUDOLABELING.SORTING)
            else:
                #No accumulation, only training with new pseudolabels
                accumulated_selection_imgA = merge_txts_and_save(os.path.join(dataset_A_path,'dataset_img.txt'),
                                                                        images_txt_A)
                accumulated_selection_pseudoA = merge_txts_and_save(os.path.join(dataset_A_path,'dataset_pseudolabels.txt'),
                                                                        psedolabels_txt_A)
                accumulated_scores_A = merge_txts_and_save(os.path.join(dataset_A_path,'filenames_and_scores.txt'),
                                                                        filenames_and_scoresA)
                accumulated_selection_imgB = merge_txts_and_save(os.path.join(dataset_B_path,'dataset_img.txt'),
                                                                        images_txt_B)
                accumulated_selection_pseudoB = merge_txts_and_save(os.path.join(dataset_B_path,'dataset_pseudolabels.txt'),
                                                                        psedolabels_txt_B)
                accumulated_scores_B = merge_txts_and_save(os.path.join(dataset_B_path,'filenames_and_scores.txt'),
                                                                        filenames_and_scoresB)

            # Save thresholding files
            np.save(os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'thresholds.npy'), cls_threshA)
            np.save(os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch),'thresholds.npy'), cls_threshB)

            total_time = time.perf_counter() - start_time
            logger.info("Accumulation done in {:.2f} s".format(total_time))

            dataset_A_source = cfg.DATASETS.TRAIN_NAME + '_A_source' + str(epoch)
            dataset_A_target = cfg.DATASETS.TRAIN_NAME + '_A_target' + str(epoch)
            dataset_B_source = cfg.DATASETS.TRAIN_NAME + '_B_source' + str(epoch)
            dataset_B_target = cfg.DATASETS.TRAIN_NAME + '_B_target' + str(epoch)

            # Alternate datasets on batch time
            # create one dataloader for the source data and another for target pseudolabels
            built_custom_dataset(cfg, source_img_datasetA, source_gt_datasetA, dataset_A_source)
            built_custom_dataset(cfg, accumulated_selection_imgA, accumulated_selection_pseudoA, dataset_A_target)
            # Train model A
            logger.info("Training Model A")
            do_train(cfg, cfg.INPUT, cfg.AUGMENTATION_A, cfg.INPUT_PSEUDO, model, weights_train_branchA, dataset_A_source, cfg.DATASETS.TEST_NAME,'a', checkpoints_A_path, epoch, cls_threshA,
                                 resume=False, dataset_pseudolabels=dataset_A_target)

            # create dataloader adding psedolabels to source dataset
            built_custom_dataset(cfg, source_img_datasetB, source_gt_datasetB, dataset_B_source)
            built_custom_dataset(cfg, accumulated_selection_imgB, accumulated_selection_pseudoB, dataset_B_target)

            # Train model B
            logger.info("Training Model B")
            do_train(cfg, cfg.INPUT2, cfg.AUGMENTATION_B, cfg.INPUT_PSEUDO, model, weights_train_branchB, dataset_B_source, cfg.DATASETS.TEST_NAME,'b', checkpoints_B_path, epoch, cls_threshB,
                                 resume=False, dataset_pseudolabels=dataset_B_target)

            # refresh weight file pointers after iteration for initial inference if there is improvement
            # The model for the next inference and training cycle is the last one obtained
            weights_inference_branchA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'checkpoints/model_final.pth')
            if not args.scratch_training:
                weights_train_branchA = weights_inference_branchA
            weights_inference_branchB = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch),'checkpoints/model_final.pth')
            if not args.scratch_training:
                weights_train_branchB = weights_inference_branchB

            if epoch < 4 and args.recompute_all_pseudolabels and cfg.SOLVER.ALTERNATE_SOURCE_PSEUDOLABELS:
                logger.info("Recompute accumulated pseudolabels and update branch A")
                recompute_pseudolabels(cfg, 'model_A', model, weights_inference_branchA, epoch, accumulated_selection_imgA,
                                       accumulated_selection_pseudoA, dataset_A_target, tgt_portion, source_priors, prior_thres,
                                       accumulated_scores_A, dataset_A_path)

                logger.info("Recompute accumulated pseudolabels and update branch B")
                recompute_pseudolabels(cfg, 'model_B', model, weights_inference_branchB, epoch, accumulated_selection_imgB,
                                       accumulated_selection_pseudoB, dataset_B_target, tgt_portion_B, source_priors, prior_thres,
                                       accumulated_scores_B, dataset_B_path)

            # Update thesholdings
            if type(tgt_portion) == np.ndarray:
                tgt_portion = np.where(tgt_portion >= max_list_tgt, max_list_tgt,
                                       tgt_portion + cfg.PSEUDOLABELING.TGT_PORT_STEP)
            else:
                tgt_portion = min(tgt_portion + cfg.PSEUDOLABELING.TGT_PORT_STEP, cfg.PSEUDOLABELING.MAX_TGT_PORT)
            if cfg.PSEUDOLABELING.INIT_TGT_PORT_B is not None:
                if type(tgt_portion_B) == np.ndarray:
                    tgt_portion_B = np.where(tgt_portion_B >= max_list_tgt_B, max_list_tgt_B,
                                       tgt_portion_B + cfg.PSEUDOLABELING.TGT_PORT_STEP_B)
                else:
                    tgt_portion_B = min(tgt_portion_B + cfg.PSEUDOLABELING.TGT_PORT_STEP_B,
                                      cfg.PSEUDOLABELING.MAX_TGT_PORT_B)
            else:
                tgt_portion_B = tgt_portion
            prior_thres = max(prior_thres-prior_relax, 0)

            # delete all datasets registered during epoch
            DatasetCatalog.remove(args.unlabeled_dataset_A_name)
            MetadataCatalog.remove(args.unlabeled_dataset_A_name)
            DatasetCatalog.remove(args.unlabeled_dataset_B_name)
            MetadataCatalog.remove(args.unlabeled_dataset_B_name)
            DatasetCatalog.remove(dataset_A_source)
            MetadataCatalog.remove(dataset_A_source)
            DatasetCatalog.remove(dataset_B_source)
            MetadataCatalog.remove(dataset_B_source)
            if cfg.SOLVER.ALTERNATE_SOURCE_PSEUDOLABELS:
                DatasetCatalog.remove(dataset_A_target)
                MetadataCatalog.remove(dataset_A_target)
                DatasetCatalog.remove(dataset_B_target)
                MetadataCatalog.remove(dataset_B_target)

    if args.ensembles:
        ensembles_folder = os.path.join(cfg.OUTPUT_DIR,'final_ensemble')
        create_folder(ensembles_folder)
        modelA = build_model(cfg)
        modelB = build_model(cfg)
        dataset_name = 'final_ensemble'
        inference_list = get_data(cfg.DATASETS.TEST_IMG_TXT)
        built_custom_dataset(cfg, cfg.DATASETS.TEST_IMG_TXT, cfg.DATASETS.TEST_GT_TXT, dataset_name)
        if args.mpt_ensemble:
            if args.no_training:
                thresA = np.load(args.thres_A)
                thresB = np.load(args.thres_B)
                print(thresA)
                print(thresB)
                cls_thres = np.where(thresA <= thresB, thresA,
                                       thresB)
            else:
                cls_thres = min(cls_threshA, cls_threshB)
            print(cls_thres)
            ensemble_on_imlist_and_save(cfg, modelA, modelB, weights_inference_branchA, weights_inference_branchB, dataset_name, inference_list, ensembles_folder, evaluation=True, mask_file=args.mask_file, thres=cls_thres)
        else:
            ensemble_on_imlist_and_save(cfg, modelA, modelB, weights_inference_branchA, weights_inference_branchB, dataset_name, inference_list, ensembles_folder, evaluation=True)


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
