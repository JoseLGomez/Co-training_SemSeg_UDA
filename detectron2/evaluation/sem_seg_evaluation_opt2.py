# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import time
import contextlib
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator

class ScoreUpdater(object):
    # only IoU are computed. accu, cls_accu, etc are ignored.
    def __init__(self, c_num, x_num, logger=None, label=None, info=None):
        self._confs = np.zeros((c_num, c_num))
        self._per_cls_iou = np.zeros(c_num)
        self._logger = logger
        self._label = label
        self._info = info
        self._num_class = c_num
        self._num_sample = x_num

    @property
    def info(self):
        return self._info

    def reset(self):
        self._start = time.time()
        self._computed = np.zeros(self._num_sample) # one-dimension
        self._confs[:] = 0

    def fast_hist(self,label, pred_label, n):
        k = (label >= 0) & (label < n)
        #print(np.bincount(n * label[k].astype(int) + pred_label[k], minlength=n ** 2))
        return np.bincount(n * label[k].astype(int) + pred_label[k], minlength=n ** 2).reshape(n, n)

    def per_class_iu(self,hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def do_updates(self, conf, i, computed=True):
        if computed:
            self._computed[i] = 1
        self._per_cls_iou = self.per_class_iu(conf)

    def update(self, pred_label, label, i, computed=True):
        conf = self.fast_hist(label, pred_label, self._num_class)
        self._confs += conf
        self.do_updates(self._confs, i, computed)
        self.scores(i)

    def scores(self, i=None, logger=None):
        x_num = self._num_sample
        ious = np.nan_to_num( self._per_cls_iou )

        logger = self._logger if logger is None else logger
        if logger is not None:
            if i is not None:
                speed = 1. * self._computed.sum() / (time.time() - self._start)
                logger.info('Done {}/{} with speed: {:.2f}/s'.format(i + 1, x_num, speed))
            name = '' if self._label is None else '{}, '.format(self._label)
            logger.info('{}mean iou: {:.2f}%'. \
                        format(name, np.mean(ious) * 100))
            with np_print_options(formatter={'float': '{:5.2f}'.format}):
                logger.info('\n{}'.format(ious * 100))

        return ious

class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(self, dataset_name, distributed=True, output_dir=None, *, num_classes=None, ignore_label=None, write_outputs=False, tgt_num=0, logger=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._write_outputs = write_outputs

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label
        self.scorer = ScoreUpdater(self._num_classes, tgt_num, logger)
        self.scorer.reset()

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        from cityscapesscripts.helpers.labels import trainId2label
        pred_output = os.path.join(self._output_dir, 'predictions')
        if not os.path.exists(pred_output):
            os.makedirs(pred_output)
        pred_colour_output = os.path.join(self._output_dir, 'colour_predictions')
        if not os.path.exists(pred_colour_output):
            os.makedirs(pred_colour_output)
        for index, batch in enumerate(zip(inputs, outputs)):
            input, output = batch
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.uint8)
            pred64 = np.array(output, dtype=np.int64) # to use it on bitcount for conf matrix
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=np.int64)
            print(self._ignore_label)
            gt[gt == self._ignore_label] = self._num_classes
            labelsid = np.arange(self._num_classes)
            gt = labelsid[gt]
            print(gt)
            print(np.unique(gt))
            self.scorer.update(pred64.flatten(), gt.flatten(), index)
            if self._write_outputs:
                file_name = input["file_name"]
                basename = os.path.splitext(os.path.basename(file_name))[0]
                pred_filename = os.path.join(pred_output, basename + '.png')
                Image.fromarray(pred).save(pred_filename)
                # colour prediction
                output = output.numpy()
                pred_colour_filename = os.path.join(pred_colour_output, basename + '.png')
                pred_colour = 255 * np.ones([output.shape[0],output.shape[1],3], dtype=np.uint8)
                for train_id, label in trainId2label.items():
                    #if label.ignoreInEval:
                    #    continue
                    #pred_colour[np.broadcast_to(output == train_id, pred_colour.shape)] = 0 #label.color
                    pred_colour[(output == train_id),0] = label.color[0]
                    pred_colour[(output == train_id),1] = label.color[1]
                    pred_colour[(output == train_id),2] = label.color[2]
                Image.fromarray(pred_colour).save(pred_colour_filename)
                #self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))


    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        '''if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))'''

        print(self._conf_matrix)
        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]

        '''if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)'''
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list

@contextlib.contextmanager
def np_print_options(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)
