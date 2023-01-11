import numpy as np
from skimage import color, io, img_as_ubyte
import os
import multiprocessing as mp


def get_data(data_list):
    with open(data_list,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    return im_list


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def compute_translation(proc_id, min_idx, max_idx, source_imgs, idxs_target, target_imgs):
    for idx in range(min_idx,max_idx):
        print('%d/%d' % (idx+1,len(source_imgs)))
        s_rgb = io.imread(source_imgs[idx][0])
        s_lab = color.rgb2lab(s_rgb[:,:,:3])
        s_mean = np.mean(s_lab, axis=tuple(range(s_lab.ndim-1)))
        s_std = np.std(s_lab, axis=tuple(range(s_lab.ndim-1)))
        t_rgb = io.imread(target_imgs[idxs_target[idx]][0])
        t_lab = color.rgb2lab(t_rgb)
        t_mean = np.mean(t_lab, axis=tuple(range(t_lab.ndim-1)))
        t_std = np.std(t_lab, axis=tuple(range(t_lab.ndim-1)))
        new_s_lab = ((s_lab - s_mean)/s_std)*t_std+t_mean
        new_s_rgb = color.lab2rgb(new_s_lab)
        path = os.path.join('/'.join(source_imgs[idx][0].split('/')[:-2]),'rgb_translated_mapillary')
        create_folder(path)
        file_name = source_imgs[idx][0].split('/')[-1]
        io.imsave(os.path.join(path, file_name), img_as_ubyte(new_s_rgb))
    

def main():
    n_workers = 8
    source = '/datatmp/Datasets/segmentation/GTA/gta5_rgb_full.txt'
    #target = '/datatmp/Datasets/segmentation/cityscapes/leftimage8bit_train.txt'
    #target = '/data1/121-1/Datasets/segmentation/TDA/fovs_images_nouab.txt'
    target = '/datatmp/Datasets/segmentation/Mapillary_Vistas/v2.0/14716_train_images_aspect_1.33.txt'
    #output_path = '/datatmp/Datasets/segmentation/GTA/rgb_translated_BDD10K_subset'
    #create_folder(output_path)
    source_imgs = get_data(source)
    target_imgs = get_data(target)
    idxs_target = np.random.random_integers(0, len(target_imgs)-1, len(source_imgs))
    print(len(np.unique(idxs_target)))
    n_filenames = len(source_imgs)
    ranges_indx = [[step*n_filenames/n_workers,(step+1)*n_filenames/n_workers] for step in range(n_workers)]
    print(ranges_indx)
    procs = []
    for proc_id in range(len(ranges_indx)):
        min_idx = int(ranges_indx[proc_id][0])
        max_idx = int(ranges_indx[proc_id][1])
        p = mp.Process(target=compute_translation,args=(proc_id, min_idx, max_idx, source_imgs, idxs_target, target_imgs))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == '__main__':
    main()