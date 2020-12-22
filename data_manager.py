import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import config as cg
from scipy import signal
import skimage.measure

class image_loader:
    def __init__(self, path_images, path_BB, path_train, path_val, path_test):
        """
        Class to load and the the images and masks of the folder.
        Args:
            path_images: (str) path to JPEGImages folder
            path_BB: (str) path to test.csv file
            path_train: (str) path to train names text
            path_val: (str) path to val names text
            path_test: (str)  path to test names text
        """

        params = self.filterFiles(path_images, cg.extension)
        self.images_names = params[0]
        self.number_total_imgs = params[1]
        self.path_images = path_images

        self.bounding_boxes = pd.read_csv(path_BB, sep=",", header=None)
        self.bounding_boxes.columns = ['filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax']
        # print("size text file: ",len(self.bounding_boxes))
        # print(" train path: ",path_train)
        self.names_train = np.loadtxt(path_train, dtype=str)
        self.names_val = np.loadtxt(path_val, dtype=str)
        self.names_test = np.loadtxt(path_test, dtype=str)
        # print("size coordinates text: ", len(self.names_train))
        # self.path_train,self.path_val, self.path_test = path_train, path_val, path_test

        self.number_img = len(self.names_train)
        self.__mode = "Train"

    def __len__(self):
        return self.number_img


    def set_train_mode(self):
        self.number_img = len(self.names_train)
        self.__mode = "Train"

    def set_val_mode(self):
        self.number_img = len(self.names_val)
        self.__mode = "Val"

    def set_test_mode(self):
        self.number_img = len(self.names_test)
        self.__mode = "Test"

    def get_name(self, index):
        if self.__mode == "Train":
            return self.names_train[index] + cg.extension
        elif self.__mode == "Val":
            return self.names_val[index] + cg.extension
        elif self.__mode == "Test":
            return self.names_test[index] + cg.extension

    def get_names(self, index, end = -1):
        if self.__mode == "Train":
            names = self.names_train[index:end]
            for i in range(len(names)):
                names[i] +=  cg.extension
            return names
        elif self.__mode == "Val":
            names = self.names_val[index:end]
            for i in range(len(names)):
                names[i] += cg.extension
            return names
        elif self.__mode == "Test":
            names = self.names_test[index:end]
            for i in range(len(names)):
                names[i] += cg.extension
            return names

    def filterFiles(self, directoryPath, extension):
        """
            This method filters the format files with the selected extension in the directory

            Args:
                directoryPath (str): relative path of the directory that contains text files
                extension (str): extension file

            Returns:
                The list of filtered files with the selected extension
        """
        # print("desired folder: ",directoryPath)
        relevant_path = directoryPath
        included_extensions = [extension]
        file_names = [file1 for file1 in os.listdir(relevant_path) if
                      any(file1.endswith(ext) for ext in included_extensions)]

        numberOfFiles = len(file_names)
        # print("number files: ",numberOfFiles)
        listParams = [file_names, numberOfFiles]
        return listParams

    def get_image(self, Index):
        """
        Method to get the image number Index, in the current set (train, val, test)
        Args:
            Index: (int) image index in ImageSets/<file>.txt

        Returns: (numpy.array)
        """
        return plt.imread(self.path_images + self.get_name(Index))

    def get_mask(self, Index):
        """
        Method to get the image mask number Index, in the current set (train, val, test)
        Args:
            Index: (int) image index in ImageSets/<file>.txt

        Returns: (numpy.array)
        """
        mask = np.zeros([cg.img_rows, cg.img_cols], dtype=cg.dtype_masks)
        for _, row in self.bounding_boxes[self.bounding_boxes.filename == self.get_name(Index)].iterrows():
            if row.cell_type == 'WBC':
                # print("WBC")
                xmin = int(row.xmin)
                xmax = int(row.xmax)
                ymin = int(row.ymin)
                ymax = int(row.ymax)
                self.colour_mask(mask, xmin, ymin, xmax, ymax)
        return mask

    def iterate(self):
        return range(self.number_imgs)

    def colour_mask(self, mask, xmin, ymin, xmax, ymax):
        mask[ymin:ymax, xmin:xmax] = 1


# data_loader = image_loader(cg.images_folder, cg.bounding_boxes, cg.train_txt, cg.val_txt, cg.test_txt)
# img = data_loader.get_mask_image(123)
# print("shape: ",img.shape)
# plt.imshow(img)


class Filter_processor:
    def __init__(self, pooling=True, loader = None):
        if loader is None:
            self.image_loader = image_loader(cg.images_folder, cg.bounding_boxes, cg.train_txt, cg.val_txt, cg.test_txt)
        else:
            self.image_loader = loader
        self.batch_size = cg.batch_size
        self.size = len(self.image_loader)
        self.end = self.size

    def evaluate(self,Filter,Mean,Var):
        """

        Args:
            Filter: list of filters, one per convolutional layer
            Mean: mean for batch normalization
            Var: var for batch normalization
        Returns: score (float)
        """
        score = 0
        samples_thus_far = 0

        if self.end % self.batch_size == 0:
            steps = np.arange(self.batch_size, self.end, self.batch_size)
        else:
            steps = np.concatenate((np.arange(self.batch_size, self.end+1, self.batch_size), np.array([self.end])))
        # print("evaluating {:d} samples".format(self.end))
        for i in steps:
            tensor = self.get_samples_img(samples_thus_far, i)#get batch

            # convolution_result = self.convolution_layer_batch(tensor,self.batch_size,Filter[0],
            #                                                   Mean[0],Var[0],samples_thus_far)
            #

            mask_output, Mean, Var = self.forward(tensor, self.batch_size,
                                  Filter, Mean, Var, samples_thus_far)
            mask = self.get_samples_mask(samples_thus_far, i)
            samples_thus_far += self.batch_size
            mask_output = self.expand_tensor(mask_output,cg.img_rows,cg.img_cols)

            score += self.score_predictions(mask,mask_output)

        return score/self.end, Mean, Var  # return average score

    def predict_img(self,filter,mean,var,images_ind):
        tensor = self.get_samples_img(images_ind, images_ind+1)  # get batch

        # convolution_result = self.convolution_layer_batch(tensor,self.batch_size,Filter[0],
        #                                                   Mean[0],Var[0],samples_thus_far)
        #

        mask_output, Mean, Var = self.forward(tensor, 1,
                                              filter, mean, var, 100)


        mask_output = self.expand_tensor(mask_output, cg.img_rows, cg.img_cols)

        return mask_output

    def get_samples_img(self, init, end):
        """
        Method to load a tensor of images starting from the image index init to image index end.
        Args:
            init (int):
            end (int):

        Returns: (ndarray) array of shape [end-init,row,col,channels]

        """
        names = self.image_loader.get_names(init,end)
        result = np.empty([len(names),cg.img_rows,cg.img_cols,cg.img_channels],dtype=cg.dtype_images)
        # print("names images: ",names[0],'\n',names[-1])
        for i in range(len(names)):
            result[i] = self.image_loader.get_image(i+init)
        return result

    def get_samples_mask(self, init, end):
        """
        Method to load a mask/ground truth target starting from the mask index init to mask index end. Mask's
        correspond to the equivalent image index. Mask is a value input
        Args:
            init (int):
            end (int):

        Returns: (ndarray) array of shape [end-init,row,col,channels]

        """
        names = self.image_loader.get_names(init, end)
        result = np.empty([len(names), cg.img_rows, cg.img_cols], dtype=cg.dtype_images)
        # print("names images: ", names[0], '\n', names[-1])
        for i in range(len(names)):
            result[i] = self.image_loader.get_mask(i + init)
        return result

    def forward(self, batch, batch_size, filters, mean, std, samples):
        """
        Method to calculate the resulting chanel of each img sample of the given batch processed forward the N
        convolutional layers. Each convolution layer is defined by a kernel, an activation function. All layer also have
        a pseudo batch normalization process, save for the last one.
        Args:

            batch (ndarray): numpy array of shape [b_size,col,row]
            filters (ndarray):
            batch_size (int):
            mean (ndarray):
            std (ndarray): 
            samples (int): number of samples used thus far to calculate the mean and batch normalization 

        Returns: batch_filtered (ndarray): numpy array of shape [b_size,col_final,row_final]
        """
        n_filters = len(filters)
        tensor = batch.copy()
        for i in range(n_filters):
            tensor = self.filter_batch(filters[i], tensor)
            if i < n_filters - 1:  # most layers
                # tensor = self.batch_forward(filters[i],tensor)
                # print("old mean: ", mean[i])
                tensor = np.maximum(0, tensor)  # activation
                new_mean = np.mean(tensor, axis=(0,1,2,3))
                new_std = np.std(tensor, axis=(0,1,2,3))
                # print("new mean: ",new_mean)
                mean[i] = self.update_value(new_mean,mean[i],batch_size,samples)  # weighted average
                std[i] = self.update_value(new_std,std[i],batch_size,samples)
                # print("mean: ", mean[i])
                if std[i] < cg.minimum_std:
                    std[i] = cg.minimum_std
                tensor = (tensor-mean[i])/std[i]  # normalization

            else:  # final layer
                tensor = np.maximum(cg.minimum_log_result, tensor)  # lower limitation
                tensor = 1/(1+np.exp(-tensor))  # sigmoid
                tensor = tensor/(np.max(tensor,axis=(1,2))[:,None,None])  # Standardization [0,1]
        return tensor, mean, std


    def filter_batch(self, filter, batch):
        """
        Args:
            filter (ndarray): filter that define the convolution layer operation. Filter of shape [kernel_x,kernel_y,
            ch_in, ch_out]
            batch (ndarray): tensor of dimensions [n_batches,tensor_row,tensor_col,tensor_channnels]

        Returns: tensor calculated for all samples in the batch. Tensor of shape [n_batches,out_row,out_col,out_ch]

        """
        n_batches = len(batch)
        # Tensor = batch.copy()
        output = []
        for i in range(n_batches):
            output += [self.convolution_layer(filter, batch[i])]
            # print("shape>: ",output[-1].shape)
        return np.array(output)



    def convolution_layer(self, filter, tensor_input):
        """
        Method that computes the tensorial convolution 2D between the corresponding filter and the input +
        and calculates the pooled output (pooling defined in the config file).
        Args:
            filter (ndarray):
        """
        # channels = self.image_loader.get_image(Index)
        h, w, ch_in = tensor_input.shape[0], tensor_input.shape[1], tensor_input.shape[2]
        h2, w2, ch_in, ch_out = filter.shape[0], filter.shape[1], filter.shape[2], filter.shape[3]
        result = np.zeros([h, w, ch_out])
        for i in range(ch_out):
            for j in range(ch_in):
                # out = signal.convolve2d(tensor_input[:, :, j], filter[:, :, j, i], mode='same', boundary='symm')
                # print("out: ",out.shape)
                # result[:, :, i] += out
                result[:, :, i] += signal.convolve2d(tensor_input[:, :, j], filter[:, :, j, i], mode='same', boundary='symm')

        result = skimage.measure.block_reduce(result,(cg.pooling_stride,cg.pooling_stride,1),np.max)#pooling
        return result

    def update_value(self, new, old, samples_new, sample_old):
        return ((sample_old)*old+samples_new*new)/(sample_old+samples_new)

    def expand_tensor(self, tensor, img_rows, img_cols):
        """
        Method that recieves a batch of images of shape [b_size,row,img] and returns a similar tensor, expanded
        to dimensiones  [b_size,img_rows,img_cols]

        Args:
            tensor:
            img_rows:
            img_cols:

        Returns:

        """
        b_size,tensor_row,tensor_col = tensor.shape[0],tensor.shape[1],tensor.shape[2]
        result = np.empty([b_size,img_rows,img_cols],dtype=cg.dtype_images)
        # print("img: \n", tensor)

        factor_x, factor_y = int(img_rows / tensor_row), int(img_cols / tensor_col)
        # print("scaling factor : {:d},  {:d}".format(factor_x,factor_y))
        for i in range(b_size):
            expanded = np.kron(tensor[i,:,:,0], np.ones((factor_x, factor_y), dtype=float))
            # print("exp: \n",expanded)

            if img_rows != factor_x * tensor_row or img_cols != factor_y * tensor_col:
                # print("shape exp before adjustment:", expanded.shape)
                adjusted_expansion = np.zeros([img_rows, img_cols])
                offset_x = int((img_rows - factor_x * tensor_row) / 2)
                offset_y = int((img_cols - factor_y * tensor_col) / 2)
                # end_x = offset_x + factor_x * tensor_row
                # end_y = offset_y + factor_y * tensor_col
                # print("limit x: {:d} : {:d}".format(offset_x, end_x))
                # print("limit y: {:d} : {:d}".format(offset_y, end_y))
                adjusted_expansion[offset_x:offset_x + factor_x * tensor_row,
                offset_y:offset_y + factor_y * tensor_col] = expanded
                expanded = adjusted_expansion
            # print("shape exp:", expanded.shape)
            # print('expanded img: \n', expanded)
            kernel = np.ones([factor_x, factor_y]) / (factor_x * factor_y)  # average kernel
            result[i] = signal.convolve2d(expanded, kernel, mode='same', boundary='symm')
            # print("resulting extrapolation: \n", result)
        return result

    def score_predictions(self, mask, mask_output):
        # print("minimum: ", mask_output.min())
        score = np.where(mask_output <= cg.minimum_score, cg.minimum_score, mask_output)#
        # print("minimum: ",score.min())
        neg_log = 1-mask_output
        neg_log = np.where(neg_log <= cg.minimum_score, cg.minimum_score, neg_log)#
        neg_log = np.log(neg_log)
        score = np.where(mask==1,np.log(score),neg_log)  # -entropy per pixel.
        return np.mean(score,axis=(0,1,2)) # we want to maximize


    def set_training_set_size(self, value):
        if value < self.size:
            self.end = value






