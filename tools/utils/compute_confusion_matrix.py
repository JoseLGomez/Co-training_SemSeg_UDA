import numpy as np
import argparse
import os
import PIL.Image as Image
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cityscapes_gt_path = '/datatmp/Datasets/segmentation/cityscapes/gtFine_trainvaltest/gtFine/train'

def args_parser():
    parser = argparse.ArgumentParser(description='Compute confusion matrix')
    parser.add_argument(
        '--input_labels',
        dest='input_labels',
        help='Txt file with labels',
        default=None,
        type=str
    )
    parser.add_argument(
        '--input_gt',
        dest='input_gt',
        help='Txt file with gt',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        help='Output directory',
        default=None,
        type=str
    )
    parser.add_argument(
        '--num_classes',
        dest='num_classes',
        help='number of classes',
        default=19,
        type=int
    )
    parser.add_argument(
        '--description',
        dest='description',
        help='Short description used in plot title and save filename',
        default='',
        type=str
    )
    args = parser.parse_args()
    return args

def plot_confusion_matrix(conf_matrix, text, save_path):
    _, ax = plt.subplots(figsize=(25,25))
    plt.rcParams.update({'font.size': 16})
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix = np.nan_to_num(conf_matrix)
    conf_matrix = np.around(conf_matrix, decimals=2)
    confd = ConfusionMatrixDisplay(conf_matrix)
    fig = confd.plot(cmap='Blues', ax=ax).figure_
    fig.suptitle('Confusion matrix %s' % (text))
    fig.savefig(os.path.join(save_path,'conf_matrix_%s.png' % (text)))


def get_data(dataset_list):
    with open(dataset_list,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    return im_list


def generate_cityscapes_gt_list(label_list):
    gt_list = []
    for label in label_list:
        label_name = label[0].split('/')[-1]
        city = label_name.split('_')[0]
        gt_filename = '_'.join(label_name.split('_')[:-1]) + '_gtFine_labelTrainIds.png'
        gt_list.append([os.path.join(cityscapes_gt_path,city,gt_filename)])
    return gt_list


def compute_confusion_matrix(label_files, gt_files, num_classes):
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    for idx, zipped_files in enumerate(zip(label_files, gt_files)):
        label_file, gt_file = zipped_files
        label = Image.open(label_file[0]).convert('L')
        label = np.asarray(label, dtype=np.int64)
        gt = Image.open(gt_file[0]).convert('L')
        gt = np.asarray(gt, dtype=np.int64)
        conf_matrix += np.bincount(
            (num_classes + 1) * label.reshape(-1) + gt.reshape(-1),
            minlength=conf_matrix.size,
        ).reshape(conf_matrix.shape)
        print('%d/%d' % (idx+1,len(label_files)))
    return conf_matrix


if __name__ == "__main__":
    args = args_parser()
    label_files = get_data(args.input_labels)
    if args.input_gt is not None:
        gt_files = get_data(args.input_labels)
    else:
        # Generate a cityscapes gt from label list names
        gt_files = generate_cityscapes_gt_list(label_files)
    conf_matrix = compute_confusion_matrix(label_files, gt_files, args.num_classes)
    plot_confusion_matrix(conf_matrix, args.description, args.output_dir)