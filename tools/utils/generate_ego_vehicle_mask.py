import numpy as np
import PIL.Image as Image


def get_data(data_list):
    with open(data_list,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    return im_list


'''gt_list = '/datatmp/Datasets/segmentation/cityscapes/gtFine_train_fullIds.txt'
data_gt = get_data(gt_list)
for i in range(len(data_gt)):
	gt = np.asarray(Image.open(data_gt[i][0]), dtype=np.uint8)
	if i == 0:
		acc_acum = np.zeros(gt.shape, dtype=np.int32)
	mask = gt == 1
	acc_acum = acc_acum + mask
	print('%d/%d' % (i+1,len(data_gt)))
print(np.unique(acc_acum))
mask = acc_acum != 0
mask = Image.fromarray(np.uint8(mask)*255)
mask.save('ego_vehicle_mask.png')'''

gt_list = '/datatmp/Datasets/segmentation/bdd100k/train_bdd10k_images.txt'
data_gt = get_data(gt_list)
for i in range(len(data_gt)):
	gt = np.asarray(Image.open(data_gt[i][0]), dtype=np.uint8)
	if gt.shape[0] > gt.shape[1]:
		print(data_gt[i][0])
		print(gt.shape)
	'''print('%d/%d' % (i+1,len(data_gt)))'''