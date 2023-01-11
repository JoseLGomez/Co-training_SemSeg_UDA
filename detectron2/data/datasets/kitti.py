# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image
import imagesize

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from .. import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_kitti_txt", "convert_to_kitti_json", "register_kitti_instances"]


def load_kitti_txt(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)

    with open(json_file, 'r') as f:
        v_imagenames = [line.rstrip() for line in f.readlines()]
        v_imagenames.sort()
        v_labelsnames = [line.replace('data_object_image_2/training/image_2',
                                      'training/label_2').replace('.png', '.txt') for line in v_imagenames]

    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
    v_valid_annos = np.asarray(meta.thing_classes)
    v_images_dict = []
    v_annos_dict = []
    for img_id in range(len(v_imagenames)):
        img_dict = {}
        img_dict['id'] = img_id
        img_dict['file_name'] = v_imagenames[img_id]
        img_dict['width'] = img_id
        width, height = imagesize.get(img_dict['file_name'])
        img_dict["height"] = height
        img_dict["width"] = width

        v_anno_dict = []
        with open(v_labelsnames[img_id], 'r') as f:
            v_lines = [line.rstrip().split(' ') for line in f.readlines()]
            v_labels = [line[0] for line in v_lines]
            v_data = [[float(el) for el in line[1:]] for line in v_lines]

        for label_id in range(len(v_labels)):
            if v_labels[label_id] in v_valid_annos:
                anno_dict = {}
                anno_dict['image_id'] = img_id
                anno_dict['file_name'] = v_labelsnames[img_id]
                anno_dict['is_crowd'] = False
                category_id = np.where(v_labels[label_id] == v_valid_annos)[0][0]
                anno_dict['category_id'] = category_id

                x1, y1, x2, y2 = v_data[label_id][3:7]
                x = x1
                y = y1
                w = (x2 - x1)
                h = (y2 - y1)
                bbox = [x,y, w, h] 
                anno_dict['bbox'] = bbox

                v_anno_dict.append(anno_dict)

        v_images_dict.append(img_dict)
        v_annos_dict.append(v_anno_dict)

    imgs_anns = list(zip(v_images_dict, v_annos_dict))
    logger.info("Loaded {} images in KITTI format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = img_dict["file_name"]
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in KITTI json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            objs.append(obj)
        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts


def register_kitti_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_kitti_txt(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="kitti", **metadata
    )


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_coco_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
