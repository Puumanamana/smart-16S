#!/usr/bin/env python
import numpy as np
from scipy.stats import nbinom,poisson
import pandas as pd

def logNB1m(sequences,threshold=1e-2):
    y = np.array([poisson.pmf(v,v.mean())
                  for v in sequences.T])
    y[y<threshold] = threshold
    y[y>1-threshold] = 1-threshold
    return -np.mean(np.log(1-y)) # returns -SUM{log(1-P[x=1])}

class Mapping:

    def __init__(self,N,ID=None,assignments={},initialize=True):
        self.id = ID
        self.Nseq = N
        self.assignments = assignments
        self.mutation = {'frequency': 0.1,
                         'strength': 0.5}
        self.hashtable = None
        self.contigency_table = None
        if initialize:
            self.initialize(N)
        self.setHashTable()
        self._n_cluster = None

    @property
    def n_cluster(self):
        return len(set(self.assignments))

    @classmethod
    def fromCSV(cls,filename='solution.csv'):
        assignments = pd.read_csv(filename,index_col=0)['cluster']
        mapping = cls(assignments.shape[0],
                      assignments=assignments,
                      initialize=False)
        return mapping

    def initialize(self,N):
        n_cluster = np.random.randint(N/2,N)
        # Make sure every cluster has at least one element
        firsts = np.random.choice(self.Nseq,n_cluster,replace=False)
        others = np.setdiff1d(range(self.Nseq),firsts)

        self.assignments = pd.Series( {i: np.random.randint(0,n_cluster)
                                       for i in others},
                                      name='cluster')
        for cluster,first in enumerate(firsts):
            self.assignments[first] = cluster
        self.assignments.index.name = 'marker'

    def setHashTable(self):
        self.hashtable = (self.assignments
                          .reset_index()
                          .groupby('cluster')['marker']
                          .apply(list))

    def setContigencyTable(self):
        tmp = pd.concat([self.assignments]*len(self.assignments),
                        axis=1)
        tmp.columns = tmp.index
        self.contigency_table = (tmp == tmp.T).astype(int)

    def evaluate(self,sequences):
        table = self.hashtable.apply(lambda x:sequences.iloc[x].values)
        self.scores = table.apply(logNB1m)
        return self.scores

    def mutate(self):
        probs_per_cluster = self.scores / self.scores.sum()
        mutation_probs = self.assignments.reset_index() 
        mutation_probs['probs'] = probs_per_cluster[mutation_probs.cluster].values

        def choose(entry):
            n_cluster = self.n_cluster
            p_else = (1-entry[2])/n_cluster
            probs = [p_else]*(1+n_cluster)
            probs[int(entry[1])] = entry[2]
            return np.random.choice(n_cluster+1,p=probs)
        
        self.assignments = mutation_probs.apply(choose,axis=1)
        self.setHashTable()

    # def splitMutate(self,mutation_probs):
    #     cluster_to_split = np.random.choice(self.hashtable.index,
    #                                         int(self.mutation['frequency']*self.n_cluster)+1,
    #                                         p=switch_probs/switch_probs.sum())
    #     markers = [np.random.choice(self.hashtable[cluster],
    #                                 int(1+self.mutation['strength']*
    #                                     self.hashtable[cluster].shape[0]))
    #                for cluster in cluster_to_mutate]

    #     for i,marker in enumerate(markers):
    #         self.assignments.loc[marker] = self.n_cluster + i

    # def mergeMutate(self):
        
        
    # def mutate(self):
    #     '''
    #     '''

    #     inv_fitness = self.scores.max() - self.scores
    #     mutation_probs = inv_fitness / inv_fitness.sum()


    #     if np.random.uniform() < .5: # Split a cluster
    #         self.splitMutate(mutation_probs)

    #     else:
                
    #     self.setHashTable()
            

    # def isTogether(self,read,reads_c):
    #     if self.contigency_table is None:
    #         self.setContigencyTable()
    #     return self.contigency_table.loc[read][reads_c].sum()
