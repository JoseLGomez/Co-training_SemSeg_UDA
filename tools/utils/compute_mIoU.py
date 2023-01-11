import os
import argparse
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.metrics import ConfusionMatrixDisplay
from cityscapesscripts.helpers.labels import trainId2label, labels


def args_parser():
    parser = argparse.ArgumentParser(description='Compute mIoU')
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        help='Output directory',
        default=None,
        type=str
    )
    parser.add_argument(
        '--predictions_file',
        dest='predictions_file',
        help='Predictions file',
        type=str
    )
    parser.add_argument(
        '--labels_file',
        dest='labels_file',
        help='Labels file',
        type=str
    )
    parser.add_argument(
        '--ignore_label',
        dest='ignore_label',
        help='Label to ignore in metrics',
        default=19,
        type=int
    )
    parser.add_argument(
        '--num_classes',
        dest='num_classes',
        help='number of classes',
        default=19,
        type=int
    )
    parser.add_argument(
        "--void_metric",
        action="store_true",
        help="counts void label in the metrics"
    )
    args = parser.parse_args()
    return args


def get_data(data_list):
    with open(data_list,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    return im_list


def print_txt_format(results_dict, output, text):
    with open(os.path.join(output,'results.txt'),"a+") as f:
        f.write('----- %s ----- \n' % text)
        for k, v in results_dict['sem_seg'].items():
            if 'IoU' in k:
                f.write('%s: %.4f \n' % (k, v))
        f.write('\n')


def compute_conf_matrix(predictions, labels, conf_matrix_acc, conf_matrix_rcall, num_classes, void_metric,
                        ignore_label):
    for idx, data in enumerate(zip(predictions, labels)):
        print('%d/%d' % (idx+1,len(labels)))
        pred = Image.open(data[0][0]).convert('L')
        pred = np.array(pred, dtype=np.int64)
        gt = Image.open(data[1][0]).convert('L')
        gt = np.array(gt, dtype=np.int64)
        if void_metric:
            gt[gt > (num_classes)] = ignore_label
        else:
            gt[gt > (num_classes)] = num_classes
        conf_matrix_acc += np.bincount(
            (num_classes + 1) * gt.reshape(-1) + pred.reshape(-1),
            minlength=conf_matrix_acc.size,).reshape(conf_matrix_acc.shape)
        conf_matrix_rcall += np.bincount(
            (num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
            minlength=conf_matrix_rcall.size,).reshape(conf_matrix_rcall.shape)
    return conf_matrix_acc, conf_matrix_rcall


def plot_confusion_matrix(conf_matrix, text, save_path):
    _, ax = plt.subplots(figsize=(25,25))
    plt.rcParams.update({'font.size': 16})
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix = np.nan_to_num(conf_matrix)
    conf_matrix = np.around(conf_matrix, decimals=2)
    confd = ConfusionMatrixDisplay(conf_matrix)
    fig = confd.plot(cmap='Blues', ax=ax).figure_
    fig.suptitle('%s Confusion matrix' % text)
    fig.savefig(os.path.join(save_path,'%s_conf_matrix.png' % text))


def compute_metrics(conf_matrix, num_classes, void_metric, class_names):
    if void_metric:
        acc = np.full(num_classes+1, np.nan, dtype=np.float)
        iou = np.full(num_classes+1, np.nan, dtype=np.float)
        tp = conf_matrix.diagonal()[:].astype(np.float)
        pos_gt = np.sum(conf_matrix[:, :], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(conf_matrix[:, :], axis=1).astype(np.float)
        acc_valid = pos_gt > -1
    else:
        acc = np.full(num_classes, np.nan, dtype=np.float)
        iou = np.full(num_classes, np.nan, dtype=np.float)
        tp = conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float)
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
    for i in range(num_classes):
        res["IoU-{}".format(class_names[i])] = 100 * iou[i]
    #res["mACC"] = 100 * macc
    #res["pACC"] = 100 * pacc
    #for i in range(num_classes):
    #    res["ACC-{}".format(class_names[i])] = 100 * acc[i]
    results = OrderedDict({"sem_seg": res})
    return results


def main():
    args = args_parser()
    predictions = get_data(args.predictions_file)
    labels = get_data(args.labels_file)
    classes = []
    for i in range(len(trainId2label)):
        if i in trainId2label:
            classes.append(trainId2label[i].name)
    conf_matrix_acc = np.zeros((args.num_classes + 1, args.num_classes + 1), dtype=np.int64)
    conf_matrix_rcall = np.zeros((args.num_classes + 1, args.num_classes + 1), dtype=np.int64)
    conf_matrix_acc, conf_matrix_rcall = compute_conf_matrix(predictions, labels, conf_matrix_acc,
                                                             conf_matrix_rcall, args.num_classes, args.void_metric,
                                                             args.ignore_label)
    results = compute_metrics(conf_matrix_acc, args.num_classes, args.void_metric, classes)
    plot_confusion_matrix(conf_matrix_acc, 'Accuracy', args.output_dir)
    plot_confusion_matrix(conf_matrix_rcall, 'Recall', args.output_dir)
    print_txt_format(results, args.output_dir, 'mIoU Scores')


if __name__ == "__main__":
    main()