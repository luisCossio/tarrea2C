# import string
# import random
# from typing import List
#
# import numpy as np

import Population
import pickle
import config as cg
import data_manager as dm

class Genetic_algotihm:
    population: Population.Population_base

    def __init__(self, Population, condition=None, iterations=100, iter_method=False):

        """

        :type Population: Populations.Population_base
        :param Population: [Population]
        :param condition: [function]
        :param iterations: [int]
        :param iter_method: [bool]
        """
        self.end_condition = condition
        self.population = Population
        self.evolution_score = []
        self.iterations = iterations
        self.end_by_time = iter_method
        self.path_save = 'models'
        if condition is None:
            assert iter_method

    def run(self):
        """

        :return:
        """
        self.population.calculate_best_score()

        if self.end_by_time:  # condition to end in a given number of iterations
            for i in range(self.iterations):
                self.population.new_generation()
                self.population.calculate_best_score()

        else:
            counter = 0
            while self.end_condition(self.population) and counter < self.iterations:
                # condition to end when a condition is reached.
                self.population.new_generation()
                self.population.calculate_best_score()
                counter += 1

            print("iteration: ", counter)

        self.population.show_answer()
        best_samples = self.population.get_best_individuals()
        average = self.population.get_average_fitness()
        best = self.population.get_best_individual()
        return best_samples, average, best

def save_element(path_save, sample, name):
    outfile = open(path_save+'/'+name, 'wb')
    pickle.dump(sample, outfile)
    outfile.close()

def load_element(path_save,name):
    file = open(path_save+'/'+name, 'rb')
    element = pickle.load(file)
    file.close()
    return element

def condition_score_95(population):
    """
    Function to end a loop given a good enough score

    :type population: Populations
    :param population: population to calculate the condition
    :return: True if condition is met (score>0.95)
    """
    return population.get_best_score() < 0.95

def main(args):
    if args.resume > 0:
        print("loading file: population_{:d}.pickle".format(args.resume))
        population = load_element(args.output_dir,'population_{:d}.pickle'.format(args.resume))
    else:
        manager_BCCD = dm.Filter_processor()
        population = Population.population_filters(manager_BCCD,cg.filters_per_layers,n_population = args.population,Mutation=args.mutation)

    if args.samples_train>-1:
        population.set_training_dataset_size(args.samples_train)

    genetic_algorithm = Genetic_algotihm(population,iterations=args.epochs)

    best_samples, average, best = genetic_algorithm.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Genetic algorithm trainnig for WBC detection')
    parser.add_argument('--population', default=25, help='population size')
    parser.add_argument('--mutation', default=0.1, help='population size')
    parser.add_argument('--epochs', default=50, help='number of total epochs to run')
    parser.add_argument('--output-dir', default='models', help='path where to save')
    parser.add_argument('--resume', default=0, help='resume from model number')
    parser.add_argument('--ask', default=10, help='ask to continue every n epochs')
    parser.add_argument('--samples-train', default=-1, help='bool to determine if we use just part of the training '
                                                              'dataset')  # to speed up training at first use just a
    # part if the training dataset. default -1 wich means use all dataset.
    args = parser.parse_args()
    print(args.model)

    main(args)

# class Namespace:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#
# use_frcnn = True
# model = 2
# args = Namespace(data_path='/home/luis/datasets/detection/detection/test',
#                  output_file='/home/luis/fruit_detection/MinneApple-master/predictions/test_{:d}.txt'.format(model),
#                  weight_file='/home/luis/fruit_detection/MinneApple-master/checkpoint/model_final_{:d}.pth'.format(model),
#                  device = 'cuda',
#                  mrcnn=not use_frcnn,
#                  frcnn=use_frcnn)
# main(args)

