import os
import multiprocessing as mp
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cityscapesscripts.helpers.labels import trainId2label, labels
dict_classes = {0:'Road', 1:'Sidewalk', 2:'Building', 3:'Wall', 4:'Fence', 5:'Pole', 6:'Traffic light', 7:'Traffic sign',
                    8:'Vegetation', 9:'Terrain', 10:'Sky', 11:'Person', 12:'Rider', 13:'Car', 14:'Truck', 15:'Bus',
                    16:'Train', 17:'Motorcycle', 18:'Bicycle', 19:'Void'}


def get_data(data_list):
    with open(data_list,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    return im_list


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


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
    Image.fromarray(pred_colour).save(filename)


def compute_label(proc_id, min_idx, max_idx, filter_class, void, labels, output_path_labels,
                  output_path_colour_labels):
    for idx in range(min_idx,max_idx):
        print('%d/%d' % (idx+1,len(labels)))
        gt = cv.imread(labels[idx][0], cv.IMREAD_GRAYSCALE)
        name = labels[idx][0].split('/')[-1]
        mask = np.ones(gt.shape)
        # check for original void label and move it to another class value
        mask[gt == 255] = 0
        gt[gt == 255] = void + 1
        for c_id in filter_class:
            # filter class per distance
            mask[gt == c_id] = 0
        gt[mask == 1] = void
        cv.imwrite(os.path.join(output_path_labels,name), gt)
        colour_label(gt, os.path.join(output_path_colour_labels,name))

def main():
    n_workers = 8
    filter_class = [11, 12]  # [11, 12 ,17, 18] # [0, 1, 2, 3, 4, 8, 9, 10] # [13, 14, 15, 16] # [5, 6, 7] #  # [11,12,13,14,15,16,17,18]  # thing objects
    void = 19
    #labels_list = '/datatmp/Datasets/segmentation/new_synthetic_dataset/2022-11-01-torstrasse-360/traffic-lights/labels.txt'
    #output_path = '/datatmp/Datasets/segmentation/new_synthetic_dataset/2022-11-01-torstrasse-360/traffic-lights/traffic_classes'
    #labels_list = '/datatmp/Datasets/segmentation/new_synthetic_dataset/2022-09-16-torstrasse/torstrasse/labels.txt'
    #output_path = '/datatmp/Datasets/segmentation/new_synthetic_dataset/2022-09-16-torstrasse/torstrasse/heavy_vehicles_classes'
    #labels_list = '/datatmp/Datasets/segmentation/new_synthetic_dataset/2022-08-19-poblenou/labels.txt'
    #output_path = '/datatmp/Datasets/segmentation/new_synthetic_dataset/2022-08-19-poblenou/heavy_vehicles_classes'
    #labels_list = '/datatmp/Datasets/segmentation/new_synthetic_dataset/2022-08-19-poblenou_terrain/labels.txt'
    #output_path = '/datatmp/Datasets/segmentation/new_synthetic_dataset/2022-08-19-poblenou_terrain/heavy_vehicles_classes'
    #labels_list = '/datatmp/Datasets/segmentation/Synscapes/img/labels_19.txt'
    #output_path = '/datatmp/Datasets/segmentation/Synscapes/img/labels_multiple_classes/heavy_vehicles_classes/'
    #labels_list = '/datatmp/Datasets/segmentation/GTA/gta5_gt_full.txt'
    #output_path = '/datatmp/Datasets/segmentation/GTA/labels_multiple_classes/heavy_vehicles_classes/'
    labels_list = '/datatmp/Datasets/segmentation/cityscapes/gtFine_val.txt'
    output_path = '/datatmp/Datasets/segmentation/cityscapes/labels_multiple_classes/val/human_classes'
    output_path_labels = os.path.join(output_path, 'labels')
    output_path_colour_labels = os.path.join(output_path, 'colour_labels')
    create_folder(output_path_labels)
    create_folder(output_path_colour_labels)
    labels = get_data(labels_list)
    n_filenames = len(labels)
    ranges_indx = [[step*n_filenames/n_workers,(step+1)*n_filenames/n_workers] for step in range(n_workers)]
    procs = []
    for proc_id in range(len(ranges_indx)):
        min_idx = int(ranges_indx[proc_id][0])
        max_idx = int(ranges_indx[proc_id][1])
        p = mp.Process(target=compute_label,args=(proc_id, min_idx, max_idx, filter_class, void, labels,
                                                  output_path_labels, output_path_colour_labels))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == '__main__':
    main()