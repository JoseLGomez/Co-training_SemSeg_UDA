import os
import numpy as np
import math
import multiprocessing as mp
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from cityscapesscripts.helpers.labels import trainId2label, labels


def get_data(data_list):
    with open(data_list,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    return im_list


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


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

    return img_pil


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
    legend = draw_legend(inference.shape[1], color_map, classes, n_lines=2)
    pred_colour = np.concatenate((pred_colour, legend))
    Image.fromarray(pred_colour).save(filename)


def apply_colour(colour, mask, image):
    image[mask,0] = colour[0]
    image[mask,1] = colour[1]
    image[mask,2] = colour[2]


target_gt_file = '/datatmp/Datasets/segmentation/cityscapes/gtFine_val.txt'
rgb_images_file = '/datatmp/Datasets/segmentation/cityscapes/leftimage8bit_val.txt'
model_predictions_file = '/data/new/Experiments/jlgomez/detectron2baselines/S+G+N/Ensemble_selftrainings_S+G+N_deepLabV3plus_crop512_batch8_90k_5000_20c_lr2e-3_batch4_4_preS+F/inference_val/custom_test/final/predictions.txt'
output_path = '/data/new/Experiments/jlgomez/detectron2baselines/S+G+N/Ensemble_selftrainings_S+G+N_deepLabV3plus_crop512_batch8_90k_5000_20c_lr2e-3_batch4_4_preS+F/Qualitative_results_iter90k'
plot_output = os.path.join(output_path,'Class_Sidewalk')
class_list = [1]
void_label = 19
res = (1024,2048)
num_classes = 19
dict_classes = {0:'Road', 1:'Sidewalk', 2:'Building', 3:'Wall', 4:'Fence', 5:'Pole', 6:'Traffic light', 7:'Traffic sign',
                8:'Vegetation', 9:'Terrain', 10:'Sky', 11:'Person', 12:'Rider', 13:'Car', 14:'Truck', 15:'Bus',
                16:'Train', 17:'Motorcycle', 18:'Bicycle', 19:'Void'}
'''dict_color_map = {0:[[128, 64, 128],[200, 128, 128]], 1:[[244, 35, 232]], 2:[[70, 70, 70]], 3:[[102, 102, 156]], 4:[[190, 153, 153]], 5:[[153, 153, 153], [128, 128, 128]], 6:[[250, 170, 30]], 7:[[220, 220, 0]],
                    8:[[107, 142, 35]], 9:[[152, 251, 152]], 10:[[70, 130, 180]], 11:[[220, 20, 60]], 12:[[255, 0, 0],[255, 0, 100]], 13:[[0, 0, 142]], 14:[[0, 0, 70]], 15:[[0, 60, 100]],
                    16:[[0, 80, 100]], 17:[[0, 0, 230]], 18:[[119, 11, 32]]}'''
dict_color_map = {1:[(0, 255, 0),(255, 0, 0), (0, 0, 255)]} #, 7:[(127, 255, 127),(255, 127, 127),(127, 127, 255)]}


def compute_plot(min_idx, max_idx, gt_files, rgb_files, pred_files, plot_output):
    for idx in range(min_idx, max_idx):
        gt = Image.open(gt_files[idx][0]).convert('L')
        gt = np.asarray(gt, dtype=np.uint8)
        pred = Image.open(pred_files[idx][0]).convert('L')
        pred = np.asarray(pred, dtype=np.uint8)
        img = Image.open(rgb_files[idx][0])
        filename = pred_files[idx][0].split('/')[-1]
        mask_colour = np.zeros([gt.shape[0],gt.shape[1],3], dtype=np.uint8)
        mask_no_void = np.zeros([gt.shape[0],gt.shape[1]])
        no_void_class = gt != void_label  # Used to remove FP from void class
        pred = pred * no_void_class
        color_map = []
        classes = []
        for class_id in class_list:
            gt_class = gt == class_id
            pred_class = pred == class_id
            wrong_mask = pred != gt
            colour = dict_color_map[class_id]
            # TP block
            tp_mask = np.logical_and(gt_class, pred_class)
            apply_colour(colour[0], tp_mask, mask_colour)
            color_map.append(colour[0])
            classes.append('TP ' + trainId2label[class_id].name)
            # FP block
            fp_mask = np.logical_and(wrong_mask, pred_class)
            apply_colour(colour[1], fp_mask, mask_colour)
            color_map.append(colour[1])
            classes.append('FP ' + trainId2label[class_id].name)
            # FN block
            fn_mask = np.logical_and(wrong_mask, gt_class)
            apply_colour(colour[2], fn_mask, mask_colour)
            color_map.append(colour[2])
            classes.append('FN ' + trainId2label[class_id].name)
            mask_no_void = np.logical_or(mask_no_void, np.logical_or(gt_class, pred_class))
        if np.count_nonzero(mask_no_void) > 2000:
            mask_no_void = np.asarray(mask_no_void*255, dtype=np.uint8)
            mask_colour = Image.fromarray(mask_colour)
            mask_colour.save(os.path.join(plot_output,'mask_' + filename))
            mask_colour.putalpha(127)
            mask_colour = np.asarray(mask_colour, dtype=np.uint8)
            mask_colour[:,:,3] = mask_colour[:,:,3]*mask_no_void
            mask_colour = Image.fromarray(mask_colour)
            img.paste(mask_colour, (0, 0), mask_colour)
            legend = draw_legend(gt.shape[1], color_map, classes, n_lines=1)
            img = get_concat_v(img, legend)
            img.save(os.path.join(plot_output,filename))
            print('%d/%d' % (idx+1,len(pred_files)))

def main():
    create_folder(output_path)
    create_folder(plot_output)
    n_workers = 8
    gt_files = get_data(target_gt_file)
    rgb_files = get_data(rgb_images_file)
    pred_files = get_data(model_predictions_file)
    n_filenames = len(pred_files)
    ranges_indx = [[step*n_filenames/n_workers,(step+1)*n_filenames/n_workers] for step in range(n_workers)]
    print(ranges_indx)
    procs = []
    for proc_id in range(len(ranges_indx)):
        min_idx = int(ranges_indx[proc_id][0])
        max_idx = int(ranges_indx[proc_id][1])
        p = mp.Process(target=compute_plot,args=(min_idx, max_idx, gt_files, rgb_files, pred_files, plot_output))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

if __name__ == '__main__':
    main()