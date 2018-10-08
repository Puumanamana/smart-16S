#!/usr/bin/env python
import numpy as np
from scipy.stats import nbinom,poisson
import pandas as pd

def logNB1m(sequences,threshold=1e-1):
    y = np.array([poisson.pmf(v,v.mean())
                  for v in sequences.T])
    y[y<threshold] = threshold
    y[y>1-threshold] = 1-threshold
    return -np.sum(np.log(1-y)) # returns -SUM{log(1-P[x=1])}

class Mapping:

    def __init__(self,N,ID=None,assignments={},initialize=True):
        self.id = ID
        self.Nseq = N
        self.assignments = assignments
        self.mutation_frequency = 0.3
        self.hashtable = None
        self.contigency_table = None
        if initialize:
            self.initialize(N)
        self.n_cluster = len(self.assignments.unique())
        self.setHashTable()

    def initialize(self,N):
        self.n_cluster = np.random.randint(N/2,N)
        # Make sure every cluster has at least one element
        firsts = np.random.choice(self.Nseq,self.n_cluster,replace=False)
        others = np.setdiff1d(range(self.Nseq),firsts)

        self.assignments = pd.Series( {i: np.random.randint(0,self.n_cluster)
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
        '''
        - Select random (non unit) clusters (proportionnaly to their likelihood)
        - Select a random marker for each selected cluster
        - Move it to a new cluster
        '''

        inv_fitness = self.scores.max() - self.scores
        switch_probs = inv_fitness / inv_fitness.sum()
        switched = np.random.choice(switch_probs.index,
                                    int(self.mutation_frequency*self.n_cluster),
                                    p=switch_probs.values)
        markers = [np.random.choice(self.hashtable[cluster])
                   for cluster in switched]
        for marker in markers:
            self.assignments[marker] = self.n_cluster
        self.n_cluster += int(len(switched)>0)

        self.setHashTable()

    def isTogether(self,read,reads_c):
        if self.contigency_table is None:
            self.setContigencyTable()
        return self.contigency_table.loc[read][reads_c].sum()
