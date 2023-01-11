# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
from torch import nn

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator


class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(self, dataset_name, distributed=True, output_dir=None, *, num_classes=None, ignore_label=None,
                 write_outputs=False, val_resize=None, plot_transparency=False, void_metric=False,
                 write_conf_maps=False):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        '''if num_classes is not None:
            self._logger.warning(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warning(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )'''
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._write_outputs = write_outputs
        self._val_resize = val_resize
        self._plot_transparency = plot_transparency
        self._write_conf_maps = write_conf_maps

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
        self._stuff_colors = meta.stuff_colors
        self._num_classes = num_classes  # Modified to metric properly void_class in cases when learned
        self._num_classes_meta = len(meta.stuff_classes)
        '''if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"'''
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label
        self._void_metric = void_metric

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._conf_matrix2 = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs, ensemble=False):
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
        if self._write_outputs:
            pred_output = os.path.join(self._output_dir, 'predictions')
            if not os.path.exists(pred_output):
                os.makedirs(pred_output)
            pred_colour_output = os.path.join(self._output_dir, 'colour_predictions')
            if not os.path.exists(pred_colour_output):
                os.makedirs(pred_colour_output)
        if self._plot_transparency:
            pred_composed_output = os.path.join(self._output_dir, 'composed_prediction')
            if not os.path.exists(pred_composed_output):
                os.makedirs(pred_composed_output)
        if self._write_conf_maps:
            conf_map_output = os.path.join(self._output_dir, 'conf_map_prediction')
            if not os.path.exists(conf_map_output):
                os.makedirs(conf_map_output)
        for input, output in zip(inputs, outputs):
            if not ensemble:
                conf_maps = output["sem_seg"]
                output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.uint8)
            pred64 = np.array(output, dtype=np.int64) # to use it on bitcount for conf matrix
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = Image.open(f).convert('L')
                if self._val_resize is not None:
                    gt = gt.resize((self._val_resize[1], self._val_resize[0]))
                gt = np.array(gt, dtype=np.int64)
            #gt[gt == self._ignore_label] = self._num_classes
            # if len(np.unique(gt)) > self._num_classes:
            if self._void_metric:
                gt[gt > (self._num_classes)] = self._num_classes
            else:
                gt[gt > (self._num_classes_meta)] = self._num_classes_meta
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * gt.reshape(-1) + pred64.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)
            '''self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred64.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)'''
            if self._write_outputs:
                file_name = input["file_name"]
                basename = os.path.splitext(os.path.basename(file_name))[0]
                pred_filename = os.path.join(pred_output, basename + '.png')
                Image.fromarray(pred).save(pred_filename)
                if not ensemble:
                    if self._write_conf_maps:  # save in uint8 to save some space still huge
                        import matplotlib.pyplot as plt
                        conf_maps = torch.unsqueeze(conf_maps, 0)
                        conf_maps = nn.Softmax2d()(conf_maps)
                        conf_maps = torch.squeeze(conf_maps)
                        #conf_maps = conf_maps.to(self._cpu_device)
                        conf_maps = conf_maps.cpu().numpy()
                        #conf_maps = np.amax(conf_maps, axis=0)
                        conf_maps = (conf_maps*100).astype(np.uint8)
                        #heatmap = (conf_maps[0, :, :]*100).to(torch.uint8)
                        #plt.imsave(os.path.join(conf_map_output, basename + '3.png'), heatmap, cmap='viridis')
                        np.save(os.path.join(conf_map_output, basename + ''), conf_maps)
                # colour prediction
                output_n = output.numpy()
                pred_colour_filename = os.path.join(pred_colour_output, basename + '.png')
                pred_colour = 255 * np.ones([output_n.shape[0],output_n.shape[1],3], dtype=np.uint8)
                for idx in range(len(self._stuff_colors)):
                    #if label.ignoreInEval:
                    #    continue
                    #pred_colour[np.broadcast_to(output == train_id, pred_colour.shape)] = 0 #label.color
                    pred_colour[(output_n == idx),0] = self._stuff_colors[idx][0]
                    pred_colour[(output_n == idx),1] = self._stuff_colors[idx][1]
                    pred_colour[(output_n == idx),2] = self._stuff_colors[idx][2]
                Image.fromarray(pred_colour).save(pred_colour_filename)
                #self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))
            if self._plot_transparency:
                file_name = input["file_name"]
                basename = os.path.splitext(os.path.basename(file_name))[0]
                pred_comp_filename = os.path.join(pred_composed_output, basename + '.png')
                img = np.array(input['image'], dtype=np.uint8)
                img = Image.fromarray(np.moveaxis(img, 0, -1))
                # colour prediction
                output_n = output.numpy()
                pred_colour = 255 * np.ones([output_n.shape[0],output_n.shape[1],3], dtype=np.uint8)
                for train_id, label in trainId2label.items():
                    pred_colour[(output_n == train_id),0] = label.color[0]
                    pred_colour[(output_n == train_id),1] = label.color[1]
                    pred_colour[(output_n == train_id),2] = label.color[2]
                pred_colour = Image.fromarray(pred_colour)
                pred_colour.putalpha(127)
                img.paste(pred_colour, (0, 0), pred_colour)
                img.save(pred_comp_filename)


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
        diff = self._num_classes - self._num_classes_meta
        if diff > 0:
            self._conf_matrix = self._conf_matrix[:-diff,:-diff]
        if self._void_metric:
            acc = np.full(self._num_classes, np.nan, dtype=np.float)
            iou = np.full(self._num_classes, np.nan, dtype=np.float)
            tp = self._conf_matrix.diagonal()[:].astype(np.float)
            pos_gt = np.sum(self._conf_matrix[:, :], axis=0).astype(np.float)
            class_weights = pos_gt / np.sum(pos_gt)
            pos_pred = np.sum(self._conf_matrix[:, :], axis=1).astype(np.float)
            acc_valid = pos_gt > -1
        else:
            acc = np.full(self._num_classes_meta, np.nan, dtype=np.float)
            iou = np.full(self._num_classes_meta, np.nan, dtype=np.float)
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
        self._logger.info(self._conf_matrix)
        return results, self._conf_matrix

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
