import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cityscapesscripts.helpers.labels import trainId2label, labels


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

dict_classes = {0:'Road', 1:'Sidewalk', 2:'Building', 3:'Wall', 4:'Fence', 5:'Pole', 6:'Traffic light', 7:'Traffic sign',
                8:'Vegetation', 9:'Terrain', 10:'Sky', 11:'Person', 12:'Rider', 13:'Car', 14:'Truck', 15:'Bus',
                16:'Train', 17:'Motorcycle', 18:'Bicycle', 19:'Void'}

labels_list = '/datatmp/Datasets/segmentation/new_synthetic_dataset/2022-11-01-torstrasse-360/traffic-lights/labels_traffic.txt'
output_list = '/datatmp/Datasets/segmentation/new_synthetic_dataset/2022-11-01-torstrasse-360/traffic-lights/labels_traffic_filtered.txt'
labels = get_data(labels_list)
filter_OR_classes = [6]
filter_AND_classes = []
with open(output_list,'w') as images_file:
    for i in range(len(labels)):
        print('%d/%d' % (i+1,len(labels)))
        name = labels[i][0].split('/')[-1]
        gt = cv.imread(labels[i][0], cv.IMREAD_GRAYSCALE)
        # filter class per distance
        class_count = np.unique(gt, return_counts=True)
        save_or = False
        if len(filter_OR_classes) > 0:
            for idx, _ in enumerate(filter_OR_classes):
                if filter_OR_classes[idx] in class_count[0] and \
                        class_count[1][np.where(filter_OR_classes[idx] == class_count[0])] > 5000:
                    save_or = True
                    break
        else:
            save_or = True
        if len(filter_AND_classes) > 0:
            for idx, _ in enumerate(filter_AND_classes):
                if filter_AND_classes[idx] in class_count[0] and \
                        class_count[1][np.where(filter_AND_classes[idx] == class_count[0])] > 5000:
                    save_and = True
                else:
                    save_and = False
                    break
        else:
            save_and = True
        if save_or and save_and:
            images_file.write(labels[i][0] + '\n')


