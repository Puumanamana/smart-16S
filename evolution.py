#!/usr/bin/env python

from model import Mapping
from time import time
import numpy as np
import pandas as pd
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
        self.populations = {i: Mapping(N_MARKER,ID=i) for i in range(N)}
        self.recombine_prob = .4
        self.mutation_rate = .1
        self.cores = 15
        self.fitnesses = []

    @timer
    def calc_fitnesses(self):
        fitnesses = []
        for p in self.populations.values():
            p.evaluate(sequences)
            fitnesses.append(p.fitness)
        self.fitnesses.append(fitnesses)
            
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

    def recombine(self,mappings):
        '''
        Mapping recombination between N individuals
        Input: mapping object with contigency table
        '''
        reads = np.arange(0,N_MARKER)

        new_clusters = {0:[]}
        new_assignments = {}
        n_clusters = 1

        for read in reads:
            c = 0
            choice = False
            while ~choice and c<n_clusters:
                reads_c = new_clusters[c]
                # isTogether() returns the number of reads in reads_c that are in the same cluster as read
                # Needs to check if reads_c is empty
                res = np.array([mapping.isTogether(read,reads_c) * mapping.fitness
                                for mapping in mappings])
                chooseYes = res.sum()
                chooseNo = np.sum([ mappings[i].fitness
                                    for i in np.where(res.astype(int)==0)[0] ])
                weights = np.array([chooseNo,chooseYes])
                choice = np.random.choice([False,True],p=weights/weights.sum())
                c += 1

                if choice:
                    new_clusters[c].append(read)
                    new_assignments[read] = c

        if ~choice:
            n_clusters += 1
            new_clusters[n_clusters] = [read]
            new_assignments[c] = n_clusters
        
        child = Mapping(N_MARKER,assignments=new_assignments)

        return child

    @timer
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
        self.mutate_generation()
        self.make_next_generation()
        print(np.max(self.fitnesses[-1]))

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

    ev = Evolution(100)
    ev.cycles(200)

    ev.display_fitness()

    
