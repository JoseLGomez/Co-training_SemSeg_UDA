import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from cityscapesscripts.helpers.labels import trainId2label


def heatmap2d(heatmap, class_name, save_path):
    fig = plt.figure()
    fig.suptitle('Heatmap class %s blurred' % (class_name))
    plt.imshow(heatmap, cmap='viridis')
    plt.colorbar()
    plt.show()
    fig.savefig(os.path.join(save_path,'heatmap_class_%s_blurred.png' % (class_name)))


categories = {}
for train_id, label in trainId2label.items():
    if train_id >= 0:
        categories[str(train_id)] = label[0]
heatmap = np.load('/datatmp/Datasets/segmentation/Synscapes/class_heatmaps_1216/priors_array.npy')
heatmap_norm = heatmap.copy()
output_dir = '/datatmp/Datasets/segmentation/Synscapes/class_heatmaps_1216'
for idx, obj_cls in enumerate(categories):
    print(" Class %s" % (categories[obj_cls]))
    heatmap[int(obj_cls),:,:] = gaussian_filter(heatmap[int(obj_cls),:,:], sigma=70)
    heatmap_norm[int(obj_cls),:,:] = heatmap[int(obj_cls),:,:] * 1/heatmap[int(obj_cls),:,:].max()
    heatmap2d(heatmap_norm[int(obj_cls),:,:], categories[obj_cls], output_dir)
np.save(os.path.join(output_dir,'priors_array_blurred.npy'), heatmap)
np.save(os.path.join(output_dir,'priors_array_blurred_norm.npy'), heatmap_norm)
