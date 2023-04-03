import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import PIL.Image as Image
from PIL import ImageFont
from PIL import ImageDraw
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import datetime
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg, add_hrnet_config
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    SemSegEvaluator_opt2,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.data.samplers import TrainingSampler, RandomClassSubsampling
from torch import nn
import torch
from contextlib import ExitStack, contextmanager
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data.datasets.generic_sem_seg_dataset import load_dataset_from_txt, load_dataset_to_inference
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.utils.logger import log_every_n_seconds
from cityscapesscripts.helpers.labels import trainId2label, labels

logger = logging.getLogger("detectron2")
test_dataset_name = None
softmax2d = nn.Softmax2d()
dict_classes = {0:'Road', 1:'Sidewalk', 2:'Building', 3:'Wall', 4:'Fence', 5:'Pole', 6:'Traffic light', 7:'Traffic sign',
                8:'Vegetatiom', 9:'Terrain', 10:'Sky', 11:'Person', 12:'Rider', 13:'Car', 14:'Truck', 15:'Bus',
                16:'Train', 17:'Motorcycle', 18:'Bicycle', 19:'Void'}


def print_txt_format(results_dict, output, iter_name):
    with open(os.path.join(output,'results.txt'),"a+") as f:
        logger.info('----- iteration: %s -----' % iter_name)
        f.write('----- iteration: %s ----- \n' % iter_name)
        for k, v in results_dict['sem_seg'].items():
            if 'IoU' in k:
                logger.info('%s: %.4f' % (k, v))
                f.write('%s: %.4f \n' % (k, v))
        logger.info('\n')
        f.write('\n')


def plot_confusion_matrix(conf_matrix, iteration, save_path):
    _, ax = plt.subplots(figsize=(25,25))
    plt.rcParams.update({'font.size': 16})
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix = np.nan_to_num(conf_matrix)
    conf_matrix = np.around(conf_matrix, decimals=2)
    confd = ConfusionMatrixDisplay(conf_matrix)
    fig = confd.plot(cmap='Blues', ax=ax).figure_
    fig.suptitle('Confusion matrix iteration %s' % (iteration))
    fig.savefig(os.path.join(save_path,'conf_matrix_iter_%s.png' % (iteration)))


def get_data(dataset_list):
    with open(dataset_list,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    return im_list


def built_custom_dataset(cfg, image_dir, gt_dir, split):
    dataset_name = cfg.DATASETS.TRAIN_NAME + split
    DatasetCatalog.register(
        dataset_name, lambda x=image_dir, y=gt_dir: load_dataset_from_txt(x, y)
    )
    if cfg.DATASETS.LABELS == 'cityscapes':
        MetadataCatalog.get(dataset_name).stuff_classes = [k.name for k in labels if k.trainId < 19 and k.trainId > -1]
        MetadataCatalog.get(dataset_name).stuff_colors = [k.color for k in labels if k.trainId < 19 and k.trainId > -1]
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
        ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
    )
    return dataset_name


def built_inference_dataset(cfg, im_list, dataset_name):
    DatasetCatalog.register(
        dataset_name, lambda x=im_list: load_dataset_to_inference(x)
    )
    if cfg.DATASETS.LABELS == 'cityscapes':
        MetadataCatalog.get(dataset_name).stuff_classes = [k.name for k in labels if k.trainId < 19 and k.trainId > -1]
        MetadataCatalog.get(dataset_name).stuff_colors = [k.color for k in labels if k.trainId < 19 and k.trainId > -1]
    elif cfg.DATASETS.LABELS == 'simple_TDA':
        dict_classes = {0:'Road', 1:'Vegetation', 2:'Sky', 3:'Fence', 4:'Pedestrian', 5:'Car', 6:'Sidewalk'}
        dict_color_map = {0:[255, 255, 153], 1:[107, 142, 35], 2:[31, 120, 180], 3:[106, 61, 154],
                      4:[123, 66, 173], 5:[115, 30, 218], 6:[244, 35, 232]}
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
        image_dir=im_list,
        evaluator_type="generic_sem_seg",
        ignore_label=255,
    )
    return dataset_name


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
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name, output_folder=output_folder)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if evaluator_type == "generic_sem_seg":
        return SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder, 
                               write_outputs=args.write_outputs, plot_transparency=args.plot_transparency,
                               write_conf_maps=args.write_conf_maps, ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                               void_metric=args.void_metric, num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, args, model, step_iter, no_eval=False):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        if no_eval:
            evaluator = None
        else:
            evaluator = get_evaluator(
                cfg, args, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name, step_iter)
            )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_test_txt(cfg, args, model, dataset_name, step_iter, no_eval=False):
    results = OrderedDict()
    dataset: List[Dict] = DatasetCatalog.get(dataset_name)
    if cfg.INPUT.VAL_RESIZE_SIZE is not None:
        mapper = DatasetMapper(cfg, is_train=False, augmentations=[T.Resize(cfg.INPUT.VAL_RESIZE_SIZE)])
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    else:
        data_loader = build_detection_test_loader(cfg, dataset_name)
    if no_eval:
        evaluator = None
    else:
        evaluator = get_evaluator(
            cfg, args, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name, str(step_iter))
        )
    results_i, conf_mat = inference_on_dataset(model, data_loader, evaluator, True)
    plot_confusion_matrix(conf_mat, step_iter, cfg.OUTPUT_DIR)
    results[dataset_name] = results_i
    print_txt_format(results_i, cfg.OUTPUT_DIR, step_iter)
    '''if comm.is_main_process():
        logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        print_csv_format(results_i)'''
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def compute_confidences_on_imlist(cfg, model, dataset_name):
    # Following the same detectron2.evaluation.inference_on_dataset function
    data_loader = build_detection_test_loader(cfg, dataset_name)
    total = len(data_loader)
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        #outputs = []
        scores_list = []
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
                output = output.cpu().numpy()
                labels = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)
                conf = np.amax(output,axis=0)
                #outputs.append([amax_output, conf])
                #confidence statistics
                aux_scores = np.zeros((num_classes+1), dtype=np.float32)
                for i in range(num_classes):
                    class_conf = conf*(labels==i)
                    class_conf[class_conf == 0] = np.nan
                    aux_scores[i] = np.nanmean(class_conf)
                score = np.nanmean(conf)
                aux_scores[-1] = score
                scores_list.append(aux_scores)
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
    return scores_list


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
    return pred_colour


def inference(cfg, model, dataset_name, img_list, output_path):
    mapper = DatasetMapper(cfg, is_train=False, augmentations=[T.Resize(cfg.INPUT.VAL_RESIZE_SIZE)])
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    total = len(data_loader)
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    output_path_pred = os.path.join(output_path, 'predictions')
    create_folder(output_path_pred)
    output_path_c_pred = os.path.join(output_path, 'coloured_predictions')
    create_folder(output_path_c_pred)
    output_path_trans_pred = os.path.join(output_path, 'coloured_predictions_transparency')
    create_folder(output_path_trans_pred)
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
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
                output = output.cpu().numpy()
                labels = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)
                #  conf = np.amax(output, axis=0)
                filename = img_list[idx][0].split('/')[-1].split('.')[0] + '.png'
                filename_c = img_list[idx][0].split('/')[-1].split('.')[0] + '_colour.png'
                Image.fromarray(labels).save(os.path.join(output_path_pred, filename))
                pred_colour = colour_label(labels, os.path.join(output_path_c_pred, filename_c))
                pred_colour.putalpha(127)
                img = Image.open(img_list[idx][0])
                img.paste(pred_colour, (0, 0), pred_colour)
                img.save(os.path.join(output_path_trans_pred, filename_c))
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


def generate_inference_mtp_pseudolabels(cfg, model, dataset_name, img_list, thres, output_path, mask_file=None):
    data_loader = build_detection_test_loader(cfg, dataset_name)
    total = len(data_loader)
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    txt_file = os.path.join(output_path,'pseudolabels_list.txt')
    pseudo_folder = os.path.join(output_path, "pseudolabels")
    create_folder(pseudo_folder)
    c_pseudo_folder = os.path.join(output_path, "color_pseudolabels")
    create_folder(c_pseudo_folder)
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        with open(txt_file, 'w') as f:
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
                    output = output.cpu().numpy()
                    labels = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)
                    conf = np.amax(output, axis=0)
                    if mask_file is not None:
                        mask = np.asarray(Image.open(mask_file).convert('L'), dtype=bool)
                        labels[mask] = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
                    for i in range(num_classes):
                        labels[(conf <= thres[i])*(labels == i)] = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
                        conf[(conf <= thres[i])*(labels == i)] = np.nan
                    filename = img_list[idx][0].split('/')[-1].split('.')[0] + '.png'
                    Image.fromarray(labels).save(os.path.join(pseudo_folder, filename))
                    f.write(os.path.join(pseudo_folder, filename) + '\n')
                    colour_label(labels, os.path.join(c_pseudo_folder, filename), dataset_name)
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


def save_class_specific_heatmap(cfg, model, dataset_name, img_list, output_path, heatmap_class):
    data_loader = build_detection_test_loader(cfg, dataset_name)
    total = len(data_loader)
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
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
                output = output.cpu().numpy()
                heatmap = output[heatmap_class, :, :]
                filename = img_list[idx][0].split('/')[-1].split('.')[0] + '.png'
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


def training_loop_mixdatasets(cfg, args, model, start_iter, max_iter, data_loader, data_loader2, storage,
                              optimizer, scheduler, periodic_checkpointer, writers, test_dataset_name):
    ''' Training loop that mixes two dataloaders to compose the final batch with the proportion specified'''
    results_list = []

    for data1, data2, iteration in zip(data_loader, data_loader2, range(start_iter, max_iter)):
        #print(data[0]['file_name'])
        #print('%s x %s' % (data[0]['height'], data[0]['width']))
        storage.iter = iteration
        data = data1+data2
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
            results = do_test_txt(cfg, args, model, test_dataset_name, iteration+1)
            results_list.append([results['sem_seg']['mIoU'],iteration])
            # Compared to "train_net.py", the test results are not dumped to EventStorage
            comm.synchronize()

        if iteration - start_iter > 5 and (
            (iteration + 1) % 20 == 0 or iteration == max_iter - 1
        ):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)


def do_train(cfg, args, model, resume=False, load_only_backbone=False):
    global test_dataset_name
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    if load_only_backbone:
        print('load_only_backbone')
        dict_model = torch.load(cfg.MODEL.WEIGHTS)
        for item in dict_model:
            if 'model' in item:
                for item2 in dict_model[item]:
                    if 'backbone' in item2 and item2 in model.state_dict():
                        print('%s: %s <---> %s' % (item2, model.state_dict()[item2].shape, dict_model['model'][item2].shape))
                        model.state_dict()[item2] = dict_model['model'][item2]
        #print(model.backbone.res5[0].conv1.weight)
        start_iter = 1
    else:
        start_iter = (
                checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
            )
        if not resume:
            start_iter = 1

    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # Data aug mapper
    if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
        if cfg.INPUT.MIXED_DATA:
            mapper = DatasetMapper(cfg, is_train=True,
                                   augmentations=build_sem_seg_train_aug(cfg.INPUT, cfg.AUGMENTATION,
                                                                         cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE),
                                   dataset2_name=cfg.INPUT.DATASET_NAME,
                                   augmentations2=build_sem_seg_train_aug2(cfg.INPUT, cfg.AUGMENTATION,
                                                                           cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE))
        else:
            mapper = DatasetMapper(cfg, is_train=True,
                                   augmentations=build_sem_seg_train_aug(cfg.INPUT, cfg.AUGMENTATION,
                                                                         cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE))
        if cfg.DATASETS.TRAIN_IMG_TXT2 is not None and cfg.DATASETS.TRAIN_GT_TXT2 is not None:
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
    else:
        mapper = None

    train_dataset_name = built_custom_dataset(cfg, cfg.DATASETS.TRAIN_IMG_TXT, cfg.DATASETS.TRAIN_GT_TXT, '_train')
    test_dataset_name = built_custom_dataset(cfg, cfg.DATASETS.TEST_IMG_TXT, cfg.DATASETS.TEST_GT_TXT, '_test')
    dataset: List[Dict] = DatasetCatalog.get(train_dataset_name)
    logger.info("Starting training from iteration {}".format(start_iter))
    #start_iter = 0
    with EventStorage(start_iter) as storage:
        '''images_path = os.path.join(cfg.OUTPUT_DIR, 'Crops', 'images')
        create_folder(images_path)
        labels_path = os.path.join(cfg.OUTPUT_DIR, 'Crops', 'labels')
        create_folder(labels_path)
        c_labels_path = os.path.join(cfg.OUTPUT_DIR, 'Crops', 'colour_labels')
        create_folder(c_labels_path)'''
        if cfg.SOLVER.ALTERNATE_SOURCE_PSEUDOLABELS:
            train2_dataset_name = built_custom_dataset(cfg, cfg.DATASETS.TRAIN_IMG_TXT2, cfg.DATASETS.TRAIN_GT_TXT2,
                                                           '_train2')
            dataset2: List[Dict] = DatasetCatalog.get(train2_dataset_name)
            if cfg.INPUT.RCS.ENABLED:
                sampler = RandomClassSubsampling(dataset, cfg, cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0])
            else:
                sampler = TrainingSampler(len(dataset))
            if cfg.INPUT2.RCS.ENABLED:
                sampler2 = RandomClassSubsampling(dataset2, cfg, cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1])
            else:
                sampler2 = TrainingSampler(len(dataset2))
            if cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0] > 0:
                data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                           total_batch_size=cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0])
            else:
                data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                           total_batch_size=1)
            if cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1] > 0:
                data_loader2 = build_detection_train_loader(cfg, dataset=dataset2, mapper=mapper2, sampler=sampler2,
                                                            total_batch_size=cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1])
            else:
                data_loader2 = build_detection_train_loader(cfg, dataset=dataset2, mapper=mapper2, sampler=sampler2,
                                                            total_batch_size=1)
            if cfg.SOLVER.ACTIVATE_CLASSMIX:
                if cfg.SOLVER.CLASSMIX_BATCH_RATIO[1] > 0:
                    data_loader_clm = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                               total_batch_size=cfg.SOLVER.CLASSMIX_BATCH_RATIO[1])
                else:
                    data_loader_clm = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                               total_batch_size=1)
                if cfg.SOLVER.CLASSMIX_BATCH_RATIO[0] > 0:
                    data_loader2_clm = build_detection_train_loader(cfg, dataset=dataset2, mapper=mapper2, sampler=sampler2,
                                                            total_batch_size=cfg.SOLVER.CLASSMIX_BATCH_RATIO[0])
                else:
                    data_loader2_clm = build_detection_train_loader(cfg, dataset=dataset2, mapper=mapper2, sampler=sampler2,
                                                            total_batch_size=1)
                training_loop_classmix_multidatasets(cfg, args, model, start_iter, max_iter, data_loader, data_loader2,
                                                     data_loader_clm, data_loader2_clm, storage, optimizer, scheduler,
                                                     periodic_checkpointer, writers, test_dataset_name)
            else:

                training_loop_mixdatasets(cfg, args, model, start_iter, max_iter, data_loader, data_loader2, storage,
                                          optimizer, scheduler, periodic_checkpointer, writers, test_dataset_name)
        else:
            if cfg.INPUT.RCS.ENABLED:
                sampler = RandomClassSubsampling(dataset, cfg, cfg.SOLVER.IMS_PER_BATCH)
            else:
                sampler = TrainingSampler(len(dataset))
            data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler,
                                                       total_batch_size=cfg.SOLVER.IMS_PER_BATCH)
            for data, iteration in zip(data_loader, range(start_iter, max_iter)):
                # debug data-aug
                if cfg.AUGMENTATION.CUTOUT:
                    for idx, _ in enumerate(data):
                        data[idx]['mask'] = data[idx]['sem_seg'] != 200 # recover mask from the CutOut
                        data[idx]['sem_seg'][data[idx]['sem_seg'] == 200] = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE # assign void class to the gt
                storage.iter = iteration

                '''if iteration < 25:
                    for sample in data:
                        np_img = sample['image'].numpy()
                        np_label = np.asarray(sample['sem_seg'], dtype=np.uint8)
                        plt_img = Image.fromarray(np.moveaxis(np_img, 0, -1))
                        plt_label = Image.fromarray(np_label)
                        filename = '%d_' % (iteration) + sample['file_name'].split('/')[-1].split('.')[0] + '.png'
                        print(filename)
                        plt_img.save(os.path.join(images_path, filename))
                        plt_label.save(os.path.join(labels_path, filename))
                        colour_label(np_label, os.path.join(c_labels_path, filename))
                else:
                    exit(-1)'''
                '''for item in data:
                    print(item['file_name'])
                    print(item['image'].shape)
                    print(item['sem_seg'].shape)
                    print(np.unique(item['image']))
                    print(np.unique(item['sem_seg']))'''
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
                    do_test_txt(cfg, args, model, test_dataset_name, iteration+1)
                    # Compared to "train_net.py", the test results are not dumped to EventStorage
                    comm.synchronize()

                if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
                ):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)


def training_loop_classmix_multidatasets(cfg, args, model, start_iter, max_iter, data_loader, data_loader2,
                                         data_loader_clm, data_loader2_clm, storage, optimizer, scheduler,
                                         periodic_checkpointer, writers, test_dataset_name):
    '''images_path = os.path.join(cfg.OUTPUT_DIR, 'Crops', 'images')
    create_folder(images_path)
    labels_path = os.path.join(cfg.OUTPUT_DIR, 'Crops', 'labels')
    create_folder(labels_path)
    c_labels_path = os.path.join(cfg.OUTPUT_DIR, 'Crops', 'colour_labels')
    create_folder(c_labels_path)'''
    # Training loop that mixes two dataloaders to compose the final batch with the proportion specified
    results_list = []
    for data1, data2, data_clm, data2_clm, iteration in zip(data_loader, data_loader2, data_loader_clm,
                                                            data_loader2_clm, range(start_iter, max_iter)):
        '''if iteration < 2:
            for sample in data2:
                np_img = sample['image'].numpy()
                np_label = np.asarray(sample['sem_seg'], dtype=np.uint8)
                plt_img = Image.fromarray(np.moveaxis(np_img, 0, -1))
                plt_label = Image.fromarray(np_label)
                filename = '%d_data2_' % (iteration) + sample['file_name'].split('/')[-1].split('.')[0] + '.png'
                print(filename)
                plt_img.save(os.path.join(images_path, filename))
                plt_label.save(os.path.join(labels_path, filename))
                colour_label(np_label, os.path.join(c_labels_path, filename))
        if iteration < 2:
            for sample in data_clm:
                np_img = sample['image'].numpy()
                np_label = np.asarray(sample['sem_seg'], dtype=np.uint8)
                plt_img = Image.fromarray(np.moveaxis(np_img, 0, -1))
                plt_label = Image.fromarray(np_label)
                filename = '%d_data_clm_' % (iteration) + sample['file_name'].split('/')[-1].split('.')[0] + '.png'
                print(filename)
                plt_img.save(os.path.join(images_path, filename))
                plt_label.save(os.path.join(labels_path, filename))
                colour_label(np_label, os.path.join(c_labels_path, filename))'''

        if cfg.SOLVER.CLASSMIX_BATCH_RATIO[0] > 0:
            data1_clm = compute_classmix(cfg, data2_clm, data1[-len(data2_clm):])
        else:
            data1_clm = []
        if cfg.SOLVER.CLASSMIX_BATCH_RATIO[1] > 0:
            data2_clm = compute_classmix(cfg, data_clm, data2[-len(data_clm):])
        else:
            data2_clm = []
        storage.iter = iteration
        data = data1[:-len(data2_clm)]+data1_clm+data2[:-len(data1_clm)]+data2_clm
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
            results = do_test_txt(cfg, args, model, test_dataset_name, iteration+1)
            results_list.append([results['sem_seg']['mIoU'],iteration])
            # Compared to "train_net.py", the test results are not dumped to EventStorage
            comm.synchronize()
        if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)
        '''if iteration < 2:
            for sample in data:
                np_img = sample['image'].numpy()
                np_label = np.asarray(sample['sem_seg'], dtype=np.uint8)
                plt_img = Image.fromarray(np.moveaxis(np_img, 0, -1))
                plt_label = Image.fromarray(np_label)
                filename = '%d_' % (iteration) + sample['file_name'].split('/')[-1].split('.')[0] + '.png'
                print(filename)
                plt_img.save(os.path.join(images_path, filename))
                plt_label.save(os.path.join(labels_path, filename))
                colour_label(np_label, os.path.join(c_labels_path, filename))
        else:
            exit(-1)'''
    return results_list


def compute_classmix(cfg, source, target):
    assert len(source) == len(target)
    if len(source) == 0:
        return []
    for idx, src_item in enumerate(source):
        if cfg.SOLVER.CLASSMIX_TGT_CLASSES is not None and len(cfg.SOLVER.CLASSMIX_TGT_CLASSES) > 0:
            cls_sel = torch.tensor(cfg.SOLVER.CLASSMIX_TGT_CLASSES, dtype=torch.int)
            #idx_classes = (src_item['sem_seg'][..., None] == cls_sel).any(-1)
            idx_classes = torch.zeros(src_item['sem_seg'].shape)
            #idx_classes = torch.tensor([], dtype=torch.int)
            for cls in cls_sel:
                idx_classes = torch.logical_or(idx_classes, src_item['sem_seg']==cls)
                #print(np.argwhere(src_item['sem_seg'] == cls).shape)
                #idx_cls = np.argwhere(src_item['sem_seg'] == cls)
                #print(idx_cls)
                #if len(idx_cls) > 0:
                #    idx_classes = torch.cat((idx_classes, idx_cls),1)
                #    print(idx_classes.shape)
        else:
            classes = torch.unique(src_item['sem_seg'])
            cls_sel=classes[torch.randperm(len(classes))][:int(len(classes)/2)]
            idx_classes = (src_item['sem_seg'][..., None] != cls_sel).all(-1)
        target[idx]['sem_seg'][idx_classes] = src_item['sem_seg'][idx_classes]
        target[idx]['image'][:, idx_classes] = src_item['image'][:, idx_classes]
    return target


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_hrnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def print_confidence_format(results_dict, iter_name, output):
    with open(os.path.join(output,'confidences.txt'),"a+") as f:
        logger.info('----- iteration: %s -----' % (iter_name))
        f.write('----- iteration: %s ----- \n' % (iter_name))
        for idx, cls in enumerate(dict_classes):
            logger.info('%s: %.4f' % (dict_classes[cls], results_dict[idx]))
            f.write('%s: %.4f \n' % (dict_classes[cls], results_dict[idx]))
        logger.info('Mean: %.4f' % (results_dict[-1]))
        f.write('Mean: %.4f \n' % (results_dict[-1]))
        f.write('\n')


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def main(args):
    global test_dataset_name
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.inference_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        iteration = cfg.MODEL.WEIGHTS.split('/')[-1].split('.')[0].split('_')[-1]
        output_folder = os.path.join(cfg.OUTPUT_DIR, "predictions", iteration)
        inference_dataset = get_data(cfg.DATASETS.TEST_IMG_TXT)
        dataset_name = built_inference_dataset(cfg, inference_dataset, 'inference')
        inference(cfg, model, dataset_name, inference_dataset, output_folder)
    elif not args.eval_only:
        distributed = comm.get_world_size() > 1
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        do_train(cfg, args, model, resume=args.resume, load_only_backbone=args.load_only_backbone)
    else:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        iteration = cfg.MODEL.WEIGHTS.split('/')[-1].split('.')[0].split('_')[-1]
        if args.compute_confidences:
            dataset_list = get_data(cfg.DATASETS.TEST_IMG_TXT)
            built_inference_dataset(cfg, dataset_list, 'inference_test')
            scores_list = compute_confidences_on_imlist(cfg, model, 'inference_test')
            scores_list = np.asarray(scores_list)
            mean_confidences = np.nanmean(scores_list, axis=0)
            print_confidence_format(mean_confidences, iteration, cfg.OUTPUT_DIR)
        if args.generate_mtp_pseudolabels:
            tgt_portion = cfg.PSEUDOLABELING.INIT_TGT_PORT
            if type(tgt_portion) == list:
                thres = np.asarray(tgt_portion, dtype=np.float32)
            else:
                raise Exception('unknown pseudolabeling thresholds on INIT_TGT_PORT')
            dataset_list = get_data(cfg.DATASETS.TEST_IMG_TXT)
            built_inference_dataset(cfg, dataset_list, 'inference_test')
            generate_inference_mtp_pseudolabels(cfg, model, 'inference_test', get_data(cfg.DATASETS.TEST_IMG_TXT), thres,
                                                cfg.OUTPUT_DIR, mask_file=None)
        elif args.generate_class_heatmap:
            print('Class %s selected to geneate heatmap' % dict_classes[args.heatmap_class])
            output_folder = os.path.join(cfg.OUTPUT_DIR, "Class_heatmap")
            create_folder(output_folder)
            dataset_list = get_data(cfg.DATASETS.TEST_IMG_TXT)
            built_inference_dataset(cfg, dataset_list, 'inference_test')
            save_class_specific_heatmap(cfg, model, 'inference_test', get_data(cfg.DATASETS.TEST_IMG_TXT),
                                                output_folder, args.heatmap_class)
            test_dataset_name = built_custom_dataset(cfg, cfg.DATASETS.TEST_IMG_TXT, cfg.DATASETS.TEST_GT_TXT, '_test')
            do_test_txt(cfg, args, model, test_dataset_name, iteration)
        else:
            test_dataset_name = built_custom_dataset(cfg, cfg.DATASETS.TEST_IMG_TXT, cfg.DATASETS.TEST_GT_TXT, '_test')
            do_test_txt(cfg, args, model, test_dataset_name, iteration)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
