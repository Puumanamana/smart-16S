#!/usr/bin/env python

from mapping import Mapping
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing.pool import Pool

np.random.seed(1234)
N_MARKER = 28
sequences = pd.read_csv('./sim1.csv',index_col=0)

def timer(fun):
    def fun_wrapper(*args,**kwargs):
        t0 = time()
        fun(*args,**kwargs)
        print('{0} - Elapsed time: {1:.3f}'.format(fun.__name__,time()-t0))
    return fun_wrapper

def evaluate(mapping):
    return mapping.evaluate(sequences)

def recombine(mappings):
    '''
    Mapping recombination between N individuals
    Input: mapping object with contigency table
    '''
    reads = np.arange(0,N_MARKER)

    new_clusters = {}
    new_assignments = {}
    n_clusters = 0

    for read in reads:
        c = 0 # iterator through "new" clusters
        selected = False
        while not selected and c<n_clusters:
            reads_c = new_clusters[c]
            # isTogether() returns the number of reads in reads_c that are in the same cluster as read
            # Needs to check if reads_c is empty
            res = np.array([mapping.isTogether(read,reads_c) * mapping.fitness
                            for mapping in mappings])
            chooseYes = res.sum()
            chooseNo = np.sum([ mappings[i].fitness
                                for i in np.where(res.astype(int)==0)[0] ])
            weights = np.array([chooseNo,chooseYes])
            selected = np.random.choice([False,True],p=weights/weights.sum())

            if selected:
                new_clusters[c].append(read)
                new_assignments[read] = c
            c += 1

        if not selected:
            new_clusters[n_clusters] = [read]
            new_assignments[read] = n_clusters
            n_clusters += 1

    new_assignments = pd.Series(new_assignments,name='cluster')
    new_assignments.index.name = 'marker'
    return new_assignments

class Evolution:

    def __init__(self,N):
        self.pop_size = N
        self.populations = {i: Mapping(N_MARKER,ID=i) for i in range(N)}
        self.recombine_prob = .5
        self.mutation_rate = .1
        self.cores = 5
        self.fitnesses = []

    @timer
    def calc_fitnesses(self):
        pool = Pool(self.cores)
        scores_list = list(map(evaluate,
                               self.populations.values()))
        pool.close()
        for i,scores in enumerate(scores_list):
            self.populations[i].scores = scores
            self.populations[i].fitness = scores.mean()

        self.fitnesses.append([p.fitness for p in self.populations.values()])

    def select_next_gen(self):
        fitnesses = np.array([ p.fitness for p in self.populations.values() ])
        fitnesses_inv = np.max(fitnesses)-fitnesses+0.0001

        recombination_probs = fitnesses / np.sum(fitnesses)
        drop_probs = fitnesses_inv / np.sum(fitnesses_inv)

        N_choose = int(self.recombine_prob*self.pop_size)

        parent1 = np.random.choice(range(self.pop_size),
                                   N_choose,
                                   p=recombination_probs,
                                   replace=False)

        parent2 = np.random.choice(range(self.pop_size),
                                   N_choose,
                                   p=recombination_probs,
                                   replace=False)

        removed = np.random.choice(range(self.pop_size),
                                   N_choose,
                                   p=drop_probs,
                                   replace=False)


        return zip(parent1,parent2),removed

    @timer
    def mutate_generation(self):
        for player in self.populations.values():
            if np.random.uniform() < self.mutation_rate:
                player.mutate()


    @timer
    def make_next_generation(self):
        parent_indices,removed = self.select_next_gen()
        parent_list = [(self.populations[i1],self.populations[i2])
                       for i1,i2 in parent_indices]
        t0 = time()
        pool = Pool(self.cores)
        assignments = pool.map(recombine,parent_list)
        pool.close()
        print('Recombination: {}'.format(time()-t0))

        for nb,ass in enumerate(assignments):
            child = Mapping(N_MARKER,
                            ID=removed[nb],
                            assignments=ass,
                            initialize=False)

            self.populations[removed[nb]] = child

    def cycle(self):
        self.calc_fitnesses()
        self.mutate_generation()
        self.make_next_generation()
        print(np.max(self.fitnesses[-1]))

    def cycles(self,n_gen,n_plots=6):
        for n in range(n_gen):
            print('Generation {}/{}'.format(n,n_gen))
            self.cycle()
            if (n+1) % 5 == 0:
                print(self.best().hashtable)
        return self

    def best(self):
       i_max = np.argmax(self.fitnesses[-1])
       return self.populations[i_max]

    def display_fitness(self):
        fig,ax = plt.subplots()
        for i,v in enumerate(ev.fitnesses):
            ax.scatter([i]*len(v),v,s=1,c='b')
        ax.plot(range(len(ev.fitnesses)),list(map(np.mean,ev.fitnesses)),label='mean',c='k')
        ax.plot(range(len(ev.fitnesses)),list(map(np.max,ev.fitnesses)),label='max',c='g')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    ev = Evolution(100)
    ev.cycles(100)

    ev.display_fitness()
