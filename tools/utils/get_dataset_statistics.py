import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import argparse
import json
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from cityscapesscripts.helpers.labels import trainId2label

UPSCALE = (1632, 1216)
#UPSCALE = (1280, 720)

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
        '--generate_json',
        help='save a json file on output_dir with json_name',
        action='store_true'
    )
    parser.add_argument(
        '--json_name',
        dest='json_name',
        help='json name to save',
        default='gt_statistics.json',
        type=str
    )
    parser.add_argument(
        '--dataset_gt_file',
        dest='dataset_gt_file',
        help='list with the png files from the ensemble A',
        default=None,
        type=str
    )
    parser.add_argument(
        '--load_json',
        dest='load_json',
        help='Json path to load',
        default=None,
        type=str
    )
    parser.add_argument(
        '--generate_splits',
        help='Activate data splits generation with classes apparition control',
        action='store_true'
    )
    parser.add_argument(
        '--classes',
        dest='classes',
        help='Classes to select',
        nargs='+',
        default=None,
        type=str
    )
    parser.add_argument(
        '--avoid_classes',
        dest='avoid_classes',
        help='Classes to avoid',
        nargs='+',
        default=None,
        type=str
    )
    parser.add_argument(
        '--dataset_img_path',
        dest='dataset_img_path',
        help='Path to dataset images to egnerate auxiliar files after classes filter',
        default=None,
        type=str
    )
    parser.add_argument(
        '--txt_name',
        dest='txt_name',
        help='json name to save',
        default='list_',
        type=str
    )
    parser.add_argument(
        '--min_pixels',
        dest='min_pixels',
        help='Minim number of pixels to filter a class on statistics',
        default=0,
        type=int
    )
    parser.add_argument(
        '--upscale_gt',
        help='upscale gt by the value UPSCALE inside the code',
        action='store_true'
    )
    parser.add_argument(
        '--generate_priors',
        help='generate heatmaps of the data',
        action='store_true'
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


def compute_statistics(args):
    images = []
    categories = {}
    summary = {}
    info = {"images":images, "categories":categories, "summary":summary}
    data = get_data(args.dataset_gt_file)
    for train_id, label in trainId2label.items():
        if train_id >= 0:
            categories[str(train_id)] = label[0]
            summary[str(train_id)] = []
    class_priors = np.zeros([len(categories), UPSCALE[1], UPSCALE[0]], dtype=np.float32)
    for i in range(len(data)):
        gt = Image.open(data[i][0]).convert('L')
        gt_resized = np.asarray(gt.resize(UPSCALE), dtype=np.uint8)
        gt = np.asarray(gt, dtype=np.uint8)
        classes = np.unique(gt, return_counts=True)
        image_dict = {  "id": i,
                    "file_name": data[i][0],
                    "classes": classes[0].tolist(),
                    "pixels": classes[1].tolist(),
                    "pixels_perc": (classes[1]/(gt.shape[1]*gt.shape[0])).tolist(),
                    "width": gt.shape[1],
                    "height": gt.shape[0]}
        images.append(image_dict)
        for idx, obj_cls in enumerate(classes[0]):
            if classes[1][idx] >= args.min_pixels:
                summary[str(obj_cls)].append(i)
            cls_idx = (gt_resized == obj_cls)
            class_priors[obj_cls,:,:][cls_idx] += 1
        print('%d/%d' % (i+1,len(data)))
    print("\n --- Summary --- \n")
    for idx, obj_cls in enumerate(categories):
        print("Class %s: %d images, %.2f%%" % (categories[obj_cls], len(summary[obj_cls]), len(summary[obj_cls])*100/len(data)))
        class_priors[int(obj_cls),:,:] = class_priors[int(obj_cls),:,:]/len(summary[obj_cls])
        if args.generate_priors:
            heatmap2d(class_priors[int(obj_cls),:,:], categories[obj_cls], args.output_dir)
    if args.generate_priors:
        np.save(os.path.join(args.output_dir,'priors_array.npy'), class_priors)
    print("Total images: %d" % (len(data)))
    return info


def generate_json(file, json_file):
    json.dump(file, open(json_file, 'w'))


def heatmap2d(heatmap, class_name, save_path):
    fig = plt.figure()
    fig.suptitle('Heatmap class %s' % (class_name))
    plt.imshow(heatmap, cmap='viridis')
    plt.colorbar()
    plt.show()
    fig.savefig(os.path.join(save_path,'heatmap_class_%s.png' % (class_name)))


def main():
    args = args_parser()
    if args.load_json is not None:
        with open(args.load_json) as json_file:
            info = json.load(json_file)
    else:
        info = compute_statistics(args)
        if args.generate_json:
            generate_json(info, os.path.join(args.output_dir,args.json_name))
    if args.generate_splits:
        if args.classes is not None:
            aux_list = []
            for obj_cls in args.classes:
                aux_list.extend(info["summary"][obj_cls])
            aux_list = np.asarray(aux_list)
            aux_list = np.unique(aux_list)
        else:
            aux_list = range(len(info["images"]))
        with open(os.path.join(args.output_dir, args.txt_name + '_gt.txt'),'w') as gt_file:
            with open(os.path.join(args.output_dir, args.txt_name + '_img.txt'),'w') as img_file:
                for idx in aux_list:
                    assert info["images"][idx]["id"] == idx
                    add_img = True
                    if args.classes is not None:
                        for obj_cls in args.classes:
                            if info["images"][idx]["pixels"][info["images"][idx]["classes"].index(int(obj_cls))] < 500:
                                add_img = False
                    if args.avoid_classes is not None:
                        for obj_cls2 in args.avoid_classes:
                            if int(obj_cls2) in info["images"][idx]["classes"] and info["images"][idx]["pixels"][info["images"][idx]["classes"].index(int(obj_cls2))] >= 500:
                                add_img = False
                    if add_img:
                        gt_file.write(info["images"][idx]["file_name"] + '\n')
                        name = info["images"][idx]["file_name"].split('/')[-1]
                        img_file.write(os.path.join(args.dataset_img_path, name) + '\n')
                        #print(info["images"][idx]["pixels"][info["images"][idx]["classes"].index(16)])


if __name__ == "__main__":
    main()