import os
import json
import numpy as np
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


num_models = 1
#json_file = '/datatmp/Datasets/segmentation/Mapillary_Vistas/v2.0/config_v1.2.json'
label_path = '/datatmp/Datasets/segmentation/new_syntethic_dataset2/colour_labels.txt'
output_path = '/datatmp/Datasets/segmentation/new_syntethic_dataset2/labels'
create_folder(output_path)
num_classes = 19
dict_classes = {0:'Road', 1:'Sidewalk', 2:'Building', 3:'Wall', 4:'Fence', 5:'Pole', 6:'Traffic light', 7:'Traffic sign',
                    8:'Vegetation', 9:'Terrain', 10:'Sky', 11:'Person', 12:'Rider', 13:'Car', 14:'Truck', 15:'Bus',
                    16:'Train', 17:'Motorcycle', 18:'Bicycle', 19:'Void'}

'''with open(json_file, "r") as infile:
    info_dict = json.load(infile)
for item in info_dict['labels']:
    print('Class %s, colour %s' % (item['name'],item['color']))'''
dict_color_map = {0:[[128, 64, 128],[200, 128, 128]], 1:[[244, 35, 232]], 2:[[70, 70, 70]], 3:[[102, 102, 156]], 4:[[190, 153, 153]], 5:[[153, 153, 153], [128, 128, 128]], 6:[[250, 170, 30]], 7:[[220, 220, 0]],
                    8:[[107, 142, 35]], 9:[[152, 251, 152]], 10:[[70, 130, 180]], 11:[[220, 20, 60]], 12:[[255, 0, 0],[255, 0, 100]], 13:[[0, 0, 142]], 14:[[0, 0, 70]], 15:[[0, 60, 100]],
                    16:[[0, 80, 100]], 17:[[0, 0, 230]], 18:[[119, 11, 32]]}
labels = get_data(label_path)
dict_res = {}
for i in range(len(labels)):
    if not os.path.exists(labels[i][0]):
        print(labels[i][0])
    else:
        c_gt = np.asarray(Image.open(labels[i][0]))
        print(np.unique(c_gt))
        if c_gt[:,:,0].shape not in dict_res:
            dict_res[c_gt[:,:,0].shape] = []
        dict_res[c_gt[:,:,0].shape].append(labels[i][0])
        mask_gt = np.ones((c_gt.shape[0],c_gt.shape[1]), dtype=np.uint8)*19
        for key in dict_color_map:
            if len(dict_color_map[key]) > 1:
                for colour in dict_color_map[key]:
                    idxs = np.where(np.all(c_gt == colour, axis=-1))
                    mask_gt[idxs] = key
            else:
                idxs = np.where(np.all(c_gt == dict_color_map[key][0], axis=-1))
                mask_gt[idxs] = key
        name = labels[i][0].split('/')[-1]
        im = Image.fromarray(mask_gt)
        im.save(os.path.join(output_path,name))
        #colour_label(mask_gt, os.path.join(output_path, 'color_' + name))
    #if i % 20 == 0:
    #    print('Found %d different resolutions' % len(dict_res.keys()))
    #print('%d/%d - %s' % (i+1, len(labels), labels[i][0]))
print(' ------ ')
print('Found %d different resolutions' % len(dict_res.keys()))
output_path = '/datatmp/Datasets/segmentation/Mapillary_Vistas/v2.0/splits/train'
create_folder(output_path)
dict_asp = {}
for key, value in dict_res.items():
    if key[1] >= 1024 and key[0] >= 512:
        aspect = round(key[1]/float(key[0]), 2)
        if aspect not in dict_asp:
            dict_asp[aspect] = []
        dict_asp[aspect] += value
    print('%s --> %d images' % (key, len(value)))
    with open(os.path.join(output_path, '%d_images_res_%dx%d.txt' % (len(value),key[1],key[0])),'w') as f:
        for img in value:
            f.write(img + '\n')
print('Found %d different aspect ratios' % len(dict_asp.keys()))
for key, value in dict_asp.items():
    print('%s --> %d images' % (key, len(value)))
    with open(os.path.join(output_path, '%d_images_aspect_%.2f.txt' % (len(value), key)),'w') as f:
        for img in value:
            f.write(img + '\n')
