import cv2 as cv
import random
import sys

def get_random_data(im_list, samples, seed=random.randrange(sys.maxsize)):
    im_list.sort()
    random.seed(seed)
    im_list = random.sample(im_list, min(len(im_list), samples))
    return im_list

def _get_files_from_txt(txt_file):
    files = []
    with open(txt_file,'r') as f:
        files = [line.rstrip().split(' ') for line in f.readlines()]
    return files

def load_dataset_from_txt(image_txt, gt_txt, num_samples=None):
    """ image_txt: txt file that contains the images with their respective absolute paths
        gt_txt: txt file that contains the gt with their respective absolute paths """
    dataset = []
    image_files = _get_files_from_txt(image_txt)
    gt_files = _get_files_from_txt(gt_txt)
    '''if num_samples is not None:
        seed = random.randrange(sys.maxsize)
        image_files = get_random_data(image_files, num_samples, seed)
        gt_files = get_random_data(gt_files, num_samples, seed)'''
    for idx, file in enumerate(image_files):
        dataset.append(
            {
                "file_name": file[0],
                "sem_seg_file_name": gt_files[idx][0]
            }
        )
    return dataset

def load_dataset_from_txt_and_merge(image_txt, image_txt2, gt_txt, gt_txt2, num_samples=None):
    """ image_txt: txt file that contains the images with their respective absolute paths
        gt_txt: txt file that contains the gt with their respective absolute paths """
    dataset = []
    image_files = _get_files_from_txt(image_txt) 
    gt_files = _get_files_from_txt(gt_txt)
    '''if num_samples is not None:
        seed = random.randrange(sys.maxsize)
        image_files = get_random_data(image_files, num_samples, seed)
        gt_files = get_random_data(gt_files, num_samples, seed)'''
    image_files = image_files + _get_files_from_txt(image_txt2)
    gt_files = gt_files + _get_files_from_txt(gt_txt2)
    for idx, file in enumerate(image_files):
        dataset.append(
            {
                "file_name": file[0],
                "sem_seg_file_name": gt_files[idx][0]
            }
        )
    return dataset

def load_dataset_to_inference(image_files, num_samples=None):
    """ image_txt: txt file that contains the images with their respective absolute paths
        gt_txt: txt file that contains the gt with their respective absolute paths """
    dataset = []

    for idx, file in enumerate(image_files):
        dataset.append(
            {
                "file_name": file[0],
            }
        )
    return dataset

def check_txts_size_integrity(image_txt, gt_txt):
    image_files = _get_files_from_txt(image_txt)
    gt_files = _get_files_from_txt(gt_txt)
    for idx, file in enumerate(image_files):
        img = cv.imread(file[0])
        label_img = cv.imread(gt_files[idx][0])
        if img.shape != label_img.shape:
            print('%s/%s: %s' % (img.shape, label_img.shape, file))

if __name__ == "__main__":
    image_txt='/datatmp/Datasets/segmentation/GTA/gta5_rgb.txt'
    gt_txt='/datatmp/Datasets/segmentation/GTA/gta5_gt_cityscapes.txt'
    check_txts_size_integrity(image_txt, gt_txt)