import numpy as np

###############################################################
#### Image loader parameters ##################################
###############################################################

text_path = "BCCD_Dataset-master/BCCD/ImageSets/Main/"
train_txt = text_path + "train.txt"
val_txt = text_path + "val.txt"
test_txt = text_path + "test.txt"
images_folder = "BCCD_Dataset-master/BCCD/JPEGImages/"
bounding_boxes = 'BCCD_Dataset-master/test.csv'
img_rows = 480
img_cols = 640
img_channels = 3
extension = '.jpg'
dtype_masks = np.float32

###############################################################
#### Image manager parameters ##################################
###############################################################

batch_size = 8
dtype_images = np.float32
pooling_stride = 2
minimum_score = 6.14421235332821e-06  # exp(-12)



###############################################################
#### Individuals parameters ###################################
###############################################################
starting_layers = 3
filters_per_layers = [2, 2, 1]# last value must always be = 1

kernel_size = 3
upper_limit = 1
lower_limit = -1
kernel_type = np.int32