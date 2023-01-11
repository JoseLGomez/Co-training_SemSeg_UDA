import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import PIL.Image as Image
from cityscapesscripts.helpers.labels import trainId2label, labels

def args_parser():
    parser = argparse.ArgumentParser(description='Compose ensemble')
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        help='Output directory',
        default=None,
        type=str
    )
    parser.add_argument(
        '--data_ensemble_A',
        dest='data_ensemble_A',
        help='list with the png files from the ensemble A',
        default=None,
        type=str
    )
    parser.add_argument(
        '--data_ensemble_B',
        dest='data_ensemble_B',
        help='list with the png files from the ensemble B',
        default=None,
        type=str
    )
    parser.add_argument(
        '--metrics_ensemble_A',
        dest='metrics_ensemble_A',
        help='list with the metrics per class from the ensemble A',
        default=None,
        type=str
    )
    parser.add_argument(
        '--metric_ensemble_B',
        dest='metric_ensemble_B',
        help='list with the metrics per class from the ensemble B',
        default=None,
        type=str
    )
    parser.add_argument(
        '--ensemble_mode',
        dest='ensemble_mode',
        help='Ensemble mode: 1) "majority" : on disagreement on pixel uses a hxh region around it to determine a consensus of the class',
        default=None,
        type=str
    )
    parser.add_argument(
        '--windows_size',
        dest='windows_size',
        help='Size of the window for majority mode hxh',
        default=3,
        type=int
    )
    args = parser.parse_args()
    return args

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def get_data(data_list):
    with open(data_list,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    return im_list

def colour_label(inference, filename):
    pred_colour = 255 * np.ones([inference.shape[0],inference.shape[1],3], dtype=np.uint8)
    for train_id, label in trainId2label.items():
        pred_colour[(inference == train_id),0] = label.color[0]
        pred_colour[(inference == train_id),1] = label.color[1]
        pred_colour[(inference == train_id),2] = label.color[2]
    Image.fromarray(pred_colour).save(filename)


def main():
    args = args_parser()
    sw = int(args.windows_size/2)
    data_ensemble_A = get_data(args.data_ensemble_A)
    data_ensemble_B = get_data(args.data_ensemble_B)
    output_dir = create_folder(os.path.join(args.output_dir,'ensemble_inference'))
    assert len(data_ensemble_A) == len(data_ensemble_B)
    for i in range(len(data_ensemble_A)):
        inference_A = np.asarray(Image.open(data_ensemble_A[i][0]), dtype=np.uint8)
        colour_label(inference_A, os.path.join(output_dir,data_ensemble_A[i][0].split('/')[-1].split('.')[0] + '_cA.png'))
        inference_B = np.asarray(Image.open(data_ensemble_B[i][0]), dtype=np.uint8)
        colour_label(inference_B, os.path.join(output_dir,data_ensemble_A[i][0].split('/')[-1].split('.')[0] + '_cB.png'))
        name = data_ensemble_A[i][0].split('/')[-1].split('.')[0] + '.png'
        assert inference_A.shape == inference_B.shape
        if args.ensemble_mode == 'majority':
            aux_pseudo = inference_A.copy()
            diff_idx = np.argwhere(inference_A != inference_B)
            for idx in diff_idx:
                y_s = idx[0] - sw
                y_e = idx[0] + sw + 1
                x_s = idx[1] - sw
                x_e = idx[1] + sw + 1
                if idx[0] == 0:
                    y_s = idx[0]
                if idx[0] == inference_A.shape[0] - 1:
                    y_e = idx[0] + 1
                if idx[1] == 0:
                    x_s = idx[1]
                if idx[1] == inference_A.shape[1] - 1:
                    x_e = idx[1] + 1
                rgA = inference_A[y_s:y_e,x_s:x_e]
                rgB = inference_B[y_s:y_e,x_s:x_e]
                votes = np.unique(np.concatenate((rgA, rgB), axis=0), return_counts=True)
                winner = np.argmax(votes[1])
                aux_pseudo[idx[0],idx[1]] = votes[0][winner]
            colour_label(aux_pseudo, os.path.join(output_dir,data_ensemble_A[i][0].split('/')[-1].split('.')[0] + '_c.png'))
            im = Image.fromarray(np.uint8(aux_pseudo))
            im.save(os.path.join(output_dir,name))



if __name__ == "__main__":
    main()