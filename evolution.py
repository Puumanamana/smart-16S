#!/usr/bin/env python

from itertools import chain
from mapping import Mapping
from time import time
import numpy as np
import pandas as pd
import scipy.stats as scy
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import matplotlib as mpl
mpl.use('TkAgg')
from multiprocessing.pool import Pool

np.random.seed(1234)
sequences = pd.read_csv('./sim1.csv',index_col=0)
N_MARKER = sequences.shape[0]
REG = 1

def timer(fun):
    def fun_wrapper(*args,**kwargs):
        t0 = time()
        try:
            best_before = args[0].best().fitness
        except:
            best_before = np.nan
        fun(*args,**kwargs)
        best_after = args[0].best().fitness
        
        print('{0} - Best fitness {1} --> {2} ({3:.3f}s)'.format(fun.__name__,
                                                                 best_before,
                                                                 best_after,
                                                                 time()-t0))
    return fun_wrapper

def evaluate(mapping):
    assignments = mapping.evaluate(sequences)
    cluster_scores = assignments.groupby('cluster')['fit'].agg("prod")
    fitness = cluster_scores.fillna(0).sum()
    return (assignments,cluster_scores,fitness)

def mutate(mapping):

    new_mapping = Mapping(N_MARKER,
                          assignments=mapping.assignments.copy(),
                          initialize=False)
    new_mapping.mutate()
    new_assignments,new_cluster_scores,new_fitness = evaluate(new_mapping)

    if mapping.fitness <= new_fitness:
        new_mapping.hashtable["scores"] = new_cluster_scores
        return (new_assignments,new_fitness,new_mapping.hashtable)
    else:
        return (mapping.assignments,mapping.fitness,mapping.hashtable)
    

def contingency2assignments(table):
    groups = [np.unique(np.where(x==1)[0]).tolist() for x in table.astype(int)]
    components = [group for group in groups if len(group) > 1]
    edges = list(chain(*[[ component[i:i+2]
                           for i in range(len(component)-1) ]
                         for component in components]))
    
    G = nx.Graph()
    G.add_nodes_from(np.arange(table.shape[0]))
    G.add_edges_from(edges)
    clusters = nx.connected_components(G)

    assignments = pd.Series([[]]*table.shape[0], name='cluster')
    assignments.index.name = 'marker'

    for i,group in enumerate(clusters):
        assignments.loc[group] = i

    return pd.DataFrame(assignments)

def recombine(mappings):

    [ mapping.setContingencyTable()
      for mapping in mappings ]

    parents_nclust = [ mapping.n_cluster for mapping in mappings ]
    ncluster_child = np.random.randint(min(parents_nclust),max(parents_nclust)+1)
    child = Mapping(N_MARKER)

    combined_parents = sum([mapping.contingency_table.values for mapping in mappings])
    
    contingency_table = (combined_parents == len(mappings)).astype(int)
                        
    child.assignments = contingency2assignments(contingency_table)

    pairs = np.argwhere((combined_parents < len(mappings))
                        & (combined_parents > 0))

    count = 0
    while child.n_cluster > ncluster_child:            
        i,j = pairs[count]

        clusters = (child.assignments.iloc[i]["cluster"],
                    child.assignments.iloc[j]["cluster"])
        
        child.assignments[child.assignments==clusters[1]] = clusters[0]
        count += 1

    return child.assignments


class Evolution:

    def __init__(self,N):
        self.pop_size = N
        self.populations = {i: Mapping(N_MARKER,ID=i) for i in range(N)}
        # Adaptive rates using direction of highest variability?
        self.recombine_prob = .25
        self.mutation_rate = .75
        self.pool = Pool(20)
        self.arity = 2
        self.metrics = []
        self.fitnesses = []
        
        self.solution = Mapping.fromCSV()

        sol_assign,sol_scores,sol_fitness = evaluate(self.solution)
        self.solution.assignments = sol_assign
        self.solution.fitness = sol_fitness
        self.solution.hashtable["scores"] = sol_scores

    def calc_fitnesses(self):
        results_list = list(self.pool.map(evaluate,
                                          self.populations.values()))
        for i,(assignments,cluster_scores,fitness) in enumerate(results_list):
            self.populations[i].assignments = assignments
            self.populations[i].hashtable["scores"] = cluster_scores
            self.populations[i].fitness = fitness
        self.fitnesses.append([p.fitness for p in self.populations.values()])
            
    def select_next_gen(self):
        fitnesses = scy.rankdata([ p.fitness for p in self.populations.values() ])

        recombination_probs = fitnesses / np.sum(fitnesses)

        N_choose = int(self.recombine_prob*self.pop_size)

        parents = np.random.choice(list(self.populations.keys()),
                                   [N_choose,self.arity],
                                   p=recombination_probs,
                                   replace=True)
        
        removed = sorted(self.populations.keys(),
                         key=lambda x: self.populations[x].fitness)[0:N_choose]

        return list(parents),removed

    def mutate_generation(self):
        pop_to_mutate = np.random.choice(self.pop_size,
                                         int(self.mutation_rate*self.pop_size),
                                         replace=False)

        mutated = list(self.pool.map(mutate,[self.populations[i] for i in pop_to_mutate]))

        for i,(assignments,fitness,hashtable) in zip(pop_to_mutate,mutated):
            self.populations[i].assignments = assignments
            self.populations[i].hashtable = hashtable
            self.populations[i].fitness = fitness
        
    def make_next_generation(self):
        parent_indices,removed = self.select_next_gen()
        parent_list = [[self.populations[i] for i in indices]
                       for indices in parent_indices]

        assignments = list(self.pool.map(recombine,parent_list))

        for nb,assignment in enumerate(assignments):
            assignment_df = pd.DataFrame(assignment)
        
            child = Mapping(N_MARKER,
                            ID=removed[nb],
                            assignments=assignment_df,
                            initialize=False)

            assignments,cluster_scores,fitness = evaluate(child)
            child.assignments = assignments
            child.hashtable["scores"] = cluster_scores
            child.fitness = fitness

            self.populations[removed[nb]] = child

    def save_metrics(self):
        truth = self.solution.assignments["cluster"].values
        pred = self.best().assignments["cluster"].values

        metrics = {'adjusted_rand_score:': sklearn.metrics.adjusted_rand_score(truth,pred),
                   'adjusted_mutual_info_score': sklearn.metrics.adjusted_mutual_info_score(truth,pred),
                   '# clusters_ratio': self.best().n_cluster / self.solution.n_cluster,
                   'fitness': self.best().fitness / self.solution.fitness }
        
        self.metrics.append(metrics)

    def cycle(self):
        self.calc_fitnesses()
        self.mutate_generation()
        print('Best individual: {}\nSolution fitness: {}'.format(
            self.best().fitness,
            self.solution.fitness))
        self.save_metrics()
        self.make_next_generation()

    def cycles(self,n_gen,n_plots=6):
        for n in range(n_gen):
            print('Generation {}/{}'.format(n,n_gen))
            self.cycle()
            if n % 5 == 0:
                print(pd.concat([self.best().hashtable.reindex(range(self.best().n_cluster)),
                                 self.solution.hashtable],
                                axis=1))

        self.pool.close()
        self.plot_metrics()
        self.display_fitness()
        #self.best().plot(sequences)
        return self

    def best(self):
       fitnesses = sorted([(i,p.fitness) for i,p in self.populations.items()],
                          reverse=True, key=lambda x:x[1])
       
       return self.populations[fitnesses[0][0]]

    def plot_metrics(self):

        metrics_df = pd.DataFrame(self.metrics)
        metrics_df = pd.melt(metrics_df.reset_index(),id_vars=["index"])
       
        sns.lineplot(x="index",y="value",hue="variable",data=metrics_df)
        plt.show()

    def display_fitness(self):
        fig,ax = plt.subplots()
        for i,v in enumerate(self.fitnesses):
            ax.scatter([i]*len(v),v,s=1,c='b')
        ax.plot(range(len(ev.fitnesses)),list(map(np.mean,ev.fitnesses)),label='mean',c='k')
        ax.plot(range(len(ev.fitnesses)),list(map(np.max,ev.fitnesses)),label='max',c='g')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    ev = Evolution(100)
    ev.cycles(300)
