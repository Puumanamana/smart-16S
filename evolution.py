#!/usr/bin/env python

from model import Mapping
from itertools import combinations
from time import time
import numpy as np
import pandas as pd
from scipy.stats import cauchy,norm
import matplotlib.pyplot as plt

np.random.seed(1234)
N_MARKER = 25
sequences = pd.read_csv('./sim1.csv',index_col=0)

def timer(fun):
    def fun_wrapper(x):
        t0 = time()
        fun(x)
        print('{0} - Elapsed time: {1:.3f}'.format(fun.__name__,time()-t0))
    return fun_wrapper

class Evolution:
      
    def __init__(self,N):
        self.pop_size = N
        self.populations = {i: Mapping(i,N_MARKER) for i in range(N)}
        self.recombine_prob = .4
        self.mutation_rate = .1
        self.cores = 15
        self.fitnesses = []

    def calc_fitnesses(self):
        fitnesses = []
        for p in self.populations.values():
            p.evaluate(sequences)
            fitnesses.append(p.fitness)
        self.fitnesses.append(fitnesses)
            
    def select_next_gen(self):
        fitnesses = np.array([ p.fitness for p in self.populations.values() ])
        fitnesses_inv = np.max(fitnesses)-fitnesses
        
        recombination_probs = fitnesses / np.sum(fitnesses)
        drop_probs = fitnesses_inv / np.sum(fitnesses_inv)

        parent1 = np.random.choice(range(self.pop_size),
                                   int(self.recombine_prob*self.pop_size),
                                   p=recombination_probs,
                                   replace=False)
        
        parent2 = np.random.choice(range(self.pop_size),
                                   int(self.recombine_prob*self.pop_size),
                                   p=recombination_probs,
                                   replace=False)
        
        removed = np.random.choice(range(self.pop_size),
                                   int(self.recombine_prob*self.pop_size),
                                   p=drop_probs,
                                   replace=False)

        
        return zip(parent1,parent2),removed

    def mutate_generation(self):
        for player in self.populations.values():
            if np.random.uniform() < self.mutation_rate:
                player.mutate()

    def recombine(self,players):
        child = Mapping()
        w = np.array([p.fitness for p in players])
        w /= w.sum()
        
        # To determine
        return child
    
    def make_next_generation(self):
        parent_indices,removed = self.select_next_gen()
        parent_list = [(self.populations[i1],self.populations[i2])
                       for i1,i2 in parent_indices]

        children = []
        for parents in parent_list:
            child = self.recombine(parents)
            children.append(child)
            
        for nb,child in enumerate(children):
            child.id = removed[nb]
            self.populations[removed[nb]] = child

    def cycle(self):
        self.calc_fitnesses()
        self.make_next_generation()
        self.mutate_generation()

    def cycles(self,n_gen,n_plots=6):
        fig,ax = plt.subplots(1,n_plots)
        for n in range(n_gen):
            print('Generation {}/{}'.format(n,n_gen))
            self.cycle()

            if (n+1)%(n_gen/n_plots) == 0:
                i_plot = int((n+1)/(n_gen/n_plots))-1
                self.demo(ax=ax[i_plot])
        return self

    def demo(self,ax=None):
        fitnesses = sorted(self.populations.items(),
                           key=lambda x: x[1].fitness,
                           reverse=True)
        best_player = fitnesses[0][1]

        best_player.plot_shoot(target=self.target,ax=ax)


    def display_fitness(self):
        fig,ax = plt.subplots()
        for i,v in enumerate(ev.fitnesses):
            ax.scatter([i]*len(v),v,s=1,c='b')
        ax.plot(range(len(ev.fitnesses)),list(map(np.mean,ev.fitnesses)),label='mean',c='k')
        ax.plot(range(len(ev.fitnesses)),list(map(np.max,ev.fitnesses)),label='max',c='g')
        plt.legend()
        plt.show()
        
    
if __name__ == '__main__':

    ev = Evolution(500)
    ev.cycles(200)

    ev.display_fitness()

    
