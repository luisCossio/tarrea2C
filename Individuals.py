import random
import numpy as np
from typing import List

from numpy.core._multiarray_umath import ndarray

import config as cg
import data_manager


class individual:
    def __init__(self):
        self.__fitness = 0

    def get_fitness(self):
        return self.__fitness

    def fitness(self):
        pass

    def set_fitness(self,Value):
        self.__fitness = Value

    def get_dna(self):
        return None


class filter_basic_individual(individual):
    __filters: ndarray

    def __init__(self, filters_per_layers, filter: np.ndarray = None):
        """
        Class that encapsulate the information of a convolutional operation defined by several filters.
        A filter is defined mostly by its field __filter = list[numpy.array]. Every numpy array in this list
        is an array of 2D convolutional filters of shape [kernel_size,kernel_size,channel_in,channel_out].
        The convolution operation is defined similarly to that employ in CNN, meaning, convolution between
        an input tensor of shape [batch_size, rows, cols, channels_in], and the filter. The resulting output is
        of shape [batch_size, rows_out, cols_out, channels_out]. Before the next filter is employ, the resulting
        tensor is subjected to the operation of pooling, RElu and batch_normalization.

        Args:
            filter (numpy.array):
        """
        super().__init__ ()
        self.filters_per_layer = filters_per_layers
        self.mean, self.var = [0] * (len(filters_per_layers) - 1), [1] * (len(filters_per_layers) - 1)
        # mean and val for batch
        # normalization. Each layer does pseudo batch normalization but the last one
        if filter is None:
            self.__filters = self.create_filters()
        else:
            self.__filters = filter




    def get_filters(self):
        return self.__filters

    def mutation(self):
        index = np.random.randint(0,len(self.__filters))
        self.mutate_layer(index)

    def mutate_layer(self, index_layer):
        shape = self.__filters[index_layer].shape
        indexes = []
        for i in range(4):
            indexes += [np.random.randint(0,shape[i])]
        self.__filters[index_layer][indexes[0],indexes[1],indexes[2],indexes[3]] = np.random.randint(cg.lower_limit,
                                                                                                     cg.upper_limit+1)

    def mutation_dna(self):
        pass

    def mutation_extension_dna(self):
        pass

    def cross_over(self, mate):
        filters = []
        proportion = np.random.rand()
        # print("random decimal: {:.2f}".format(proportion))
        filters2 = mate.get_filters()
        for i,filter in enumerate(self.__filters):
            new_filter = filter.copy()
            in_channels, out_channels = new_filter.shape[2], new_filter.shape[3]
            index = int(proportion*in_channels*out_channels)
            # print("shape: {:d}, {:d}".format(in_channels, out_channels))
            # print("Index {:d}/{:d}".format(index,in_channels*out_channels))
            for j in range(index,in_channels*out_channels):
                # print("positions i,j: {:d}, {:d}".format(int(j/out_channels),j%out_channels))
                new_filter[:,:,int(j/out_channels),j%out_channels] = filters2[i][:,:,int(j/out_channels),j%out_channels]
            filters += [new_filter]
        return filter_basic_individual(cg.filters_per_layers,filters)


    def fitness(self, data_manager) -> None:
        """
        Method to calculate fitness and set it in the individual
        Args:
            data_manager (data_manager.Filter_processor):

        """
        fitness,mean,var = data_manager.evaluate(self.__filters,self.mean,self.var)
        self.set_fitness(fitness)
        self.mean = mean
        self.var = var
        # return result

    def create_filters(self):
        filters = []
        for i,size_in in enumerate([cg.img_channels]+self.filters_per_layer[:-1]):
            filters += [np.random.randint(cg.lower_limit, cg.upper_limit + 1, [cg.kernel_size, cg.kernel_size,
                                                                             size_in,
                                                                             self.filters_per_layer[i]], dtype=cg.kernel_type)]

        return filters

