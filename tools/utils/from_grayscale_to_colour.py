import os
import numpy as np
import multiprocessing as mp
import math
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


def colour_label_cityscapes(inference, filename):
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
    pred_colour = Image.fromarray(pred_colour)
    pred_colour.save(filename)


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

    return np.asarray(img_pil)


def colour_label(inference, filename, dict_color_map, dict_classes):
    pred_colour = 255 * np.ones([inference.shape[0],inference.shape[1],3], dtype=np.uint8)
    for cls in dict_color_map:
        pred_colour[(inference == cls),0] = dict_color_map[cls][0]
        pred_colour[(inference == cls),1] = dict_color_map[cls][1]
        pred_colour[(inference == cls),2] = dict_color_map[cls][2]
    color_map = []
    classes = []
    for i in range(len(dict_classes)):
        color_map.append(tuple(dict_color_map[i]))
        classes.append(dict_classes[i])
    legend = draw_legend(inference.shape[1], color_map, classes, n_lines=1)
    pred_colour = np.concatenate((pred_colour, legend))
    pred_colour = Image.fromarray(pred_colour)
    pred_colour.save(filename)


def loop_call(min_idx, max_idx, labels, output_path, dict_color_map=None, dict_classes=None):
    for idx in range(min_idx,max_idx):
        name = labels[idx][0].split('/')[-1]
        img = np.asarray(Image.open(labels[idx][0]))
        if dict_color_map is not None:
            colour_label(img, os.path.join(output_path,name), dict_color_map, dict_classes)
        else:
            colour_label_cityscapes(img, os.path.join(output_path,name))
        print('%d/%d' % (idx+1,len(labels)))


def main():
    n_workers = 8
    label_path = '/data/new/Experiments/jlgomez/Test_german/baseline_translated_30K_batch8/predictions/final/predictions.txt'
    output_path = '/data/new/Experiments/jlgomez/Test_german/baseline_translated_30K_batch8/predictions/final/better_colour'
    dict_color_map = {0:[153,204,255], 1:[255,204,229], 2:[127,127,127], 3:[0, 204, 102], 4:[255, 0, 0], 5:[0, 0, 0]}
    dict_classes = {0:'Sky', 1:'Ground', 2:'Small Rocks', 3:'Vegetation', 4:'Car', 5:'Unlabeled'}
    create_folder(output_path)
    labels = get_data(label_path)
    n_filenames = len(labels)
    ranges_indx = [[step*n_filenames/n_workers,(step+1)*n_filenames/n_workers] for step in range(n_workers)]
    print(ranges_indx)
    procs = []
    for proc_id in range(len(ranges_indx)):
        min_idx = int(ranges_indx[proc_id][0])
        max_idx = int(ranges_indx[proc_id][1])
        p = mp.Process(target=loop_call,args=(min_idx, max_idx, labels, output_path, dict_color_map, dict_classes))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == '__main__':
    main()