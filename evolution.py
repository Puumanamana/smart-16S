#!/usr/bin/env python

from mapping import Mapping
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing.pool import Pool

np.random.seed(1234)
sequences = pd.read_csv('./sim1.csv',index_col=0)
N_MARKER = sequences.shape[0]
REG = 0.1

def timer(fun):
    def fun_wrapper(*args,**kwargs):
        t0 = time()
        fun(*args,**kwargs)
        print('{0} - Elapsed time: {1:.3f}'.format(fun.__name__,time()-t0))
    return fun_wrapper

def evaluate(mapping):
    scores = mapping.evaluate(sequences)
    scores_reg = scores - REG * len(scores)
    return scores_reg.apply(lambda x:max(x,0.0001))

def contingency2assignments(table):
    groups = [np.unique(np.where(x==1)[0]) for x in table.astype(int)]
    assignments = pd.Series([[]]*table.shape[0],name='cluster')
    assignments.index.name = 'marker'
    
    for i,group in enumerate(groups):
        for marker in list(group):
            assignments[marker].append(i)
    return assignments.apply(np.random.choice)

def recombine(mappings):

    [ mapping.setContingencyTable()
      for mapping in mappings ]
    
    contingency_prob = np.mean([mapping.contingency_table
                               for mapping in mappings],
                              axis=0)
    
    def choose(prob):
        return np.random.uniform() < prob
    
    table = np.array(list(
        map(choose,contingency_prob)
    ))
    
    assignments_child = contingency2assignments(table)

    return assignments_child



class Evolution:

    def __init__(self,N):
        self.pop_size = N
        self.populations = {i: Mapping(N_MARKER,ID=i) for i in range(N)}
        self.recombine_prob = .3
        self.mutation_rate = .3
        self.cores = 15
        self.fitnesses = []
        self.solution = Mapping.fromCSV()
        self.solution.fitness = self.solution.evaluate(sequences).mean()

    @timer
    def calc_fitnesses(self):
        pool = Pool(self.cores)
        scores_list = list(pool.map(evaluate,
                               self.populations.values()))
        pool.close()
        for i,scores in enumerate(scores_list):
            self.populations[i].scores = scores
            self.populations[i].fitness = scores.mean()

        self.fitnesses.append([p.fitness for p in self.populations.values()])

    def select_next_gen(self):
        fitnesses = np.array([ p.fitness for p in self.populations.values() ])
        # fitnesses_inv = np.max(fitnesses)-fitnesses+0.0001
        # drop_probs = fitnesses_inv / np.sum(fitnesses_inv)

        recombination_probs = fitnesses / np.sum(fitnesses)

        N_choose = int(self.recombine_prob*self.pop_size)

        parent1 = np.random.choice(range(self.pop_size),
                                   N_choose,
                                   p=recombination_probs,
                                   replace=False)

        parent2 = np.random.choice(range(self.pop_size),
                                   N_choose,
                                   p=recombination_probs,
                                   replace=False)
        removed = sorted(range(self.pop_size),
                         key=lambda x: self.populations[x].fitness)[0:N_choose]
        # removed = np.random.choice(range(self.pop_size),
        #                            N_choose,
        #                            p=drop_probs,
        #                            replace=False)


        return zip(parent1,parent2),removed

    @timer
    def mutate_generation(self):
        for mapping in self.populations.values():
            if np.random.uniform() < self.mutation_rate:
                assignments = mapping.assignments.copy()
                mapping.mutate()
                
                if mapping.evaluate(sequences).mean() < mapping.fitness:
                    mapping.assignments = assignments
                    mapping.setHashTable()

    @timer
    def make_next_generation(self):
        parent_indices,removed = self.select_next_gen()
        parent_list = [(self.populations[i1],self.populations[i2])
                       for i1,i2 in parent_indices]
        t0 = time()
        pool = Pool(self.cores)
        assignments = list(pool.map(recombine,parent_list))
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
        print('Best individual: {}\nSolution fitness: {}'.format(
            np.max(self.fitnesses[-1]),
            self.solution.fitness))

    def cycles(self,n_gen,n_plots=6):
        for n in range(n_gen):
            print('Generation {}/{}'.format(n,n_gen))
            self.cycle()
            if n % 3 == 0:
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
