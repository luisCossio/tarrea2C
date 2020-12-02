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
        if condition is None:
            assert iter_method

    def run(self):
        """

        :return:
        """
        score = self.population.calculate_best_score()
        print("initial score: ", score)
        if self.end_by_time:  # condition to end in a given number of iterations
            for i in range(self.iterations):
                self.population.new_generation()
                score = self.population.calculate_best_score()
                if i % 5 == 0:
                    print("iteration number: ", i)
                    print("best score: ", score)
                    save_element('models', 'population_{:d}.pickle'.format(i+1),
                                 self.population)

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


def save_element(path_save, name, sample):
    """

    Args:
        name (str):
        path_save (str):
        sample (Any)
    """
    outfile = open(path_save + '/' + name, 'wb')
    pickle.dump(sample, outfile)
    outfile.close()


def load_element(path_save, name):
    file = open(path_save + '/' + name, 'rb')
    element = pickle.load(file)
    file.close()
    return element


def main(args):
    if int(args.resume) > 0:
        print("loading file: population_{:d}.pickle".format(int(args.resume)))
        population = load_element(args.output_dir, 'population_{:d}.pickle'.format(args.resume))
    else:
        manager_BCCD = dm.Filter_processor()
        population = Population.population_filters(manager_BCCD, cg.filters_per_layers, n_population=int(args.population),
                                                   Mutation=float(args.mutation))

    if int(args.samples_train) > -1:
        population.set_training_dataset_size(int(args.samples_train))

    genetic_algorithm = Genetic_algotihm(population, iterations=int(args.epochs), iter_method=True)

    best_samples, average, best = genetic_algorithm.run()
    save_element(args.output_dir, 'population_{:d}.pickle'.format(int(args.epochs)), genetic_algorithm.population)
    print("best: ", best.get_fitness())
    print("averge result: ", average)
    for i in range(len(best_samples)):
        print("winner results is: ", best_samples[i].get_fitness())
    print("averge result: ", average)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Genetic algorithm trainnig for WBC detection')
    parser.add_argument('--population', default=25, help='population size',type=int)
    parser.add_argument('--mutation', default=0.1, help='population size', type=float)
    parser.add_argument('--epochs', default=50, help='number of total epochs to run',type=int)
    parser.add_argument('--output-dir', default='models', help='path where to save')
    parser.add_argument('--resume', default=0, help='resume from model number',type=int)
    # parser.add_argument('--ask', default=10, help='ask to continue every n epochs')
    parser.add_argument('--samples-train', default=-1, help='bool to determine if we use just part of the training '
                                                              'dataset',type=int)  # to speed up training at first use just a
    # part if the training dataset. default -1 wich means use all dataset.
    args = parser.parse_args()

    main(args)


# class Namespace:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#
#
# args = Namespace(population=10,
#                  mutation=0.1,
#                  epochs=5,
#                  output_dir='models',
#                  resume=0,
#                  samples_train=25)
# main(args)
