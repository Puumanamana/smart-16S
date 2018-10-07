#!/usr/bin/env python
import numpy as np
from scipy.stats import nbinom
import pandas as pd

def convert_params(mu, sigma):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy 
    """
    mu = max(1,mu)
    var = max(1,sigma**2)
    
    p = (var - mu) / var
    r = mu**2 / (var-mu)
    
    return r, 1-p

class Mapping:

    def __init__(self,ID,N,assignments={}):
        self.id = ID
        self.Nseq = N
        self.assignments = assignments
        self.scores = np.array([])
        self.mutation_frequency = 0.1
        self.initialize(N)

    def initialize(self,N):
        self.n_cluster = np.random.randint(1,N/4)
        self.assignments = pd.Series( {i: np.random.randint(1,self.n_cluster)
                                       for i in range(N)},
                                      name=['cluster'])
        self.assignments.index.name = 'marker'
        self.setHashTable()

    def setHashtable(self):
        if self.hashtable is None:
            self.hashtable = (self.assignments
                              .reset_index()
                              .groupby('cluster')['marker']
                              .apply(list))
        
    def evaluate(self,sequences):
        self.scores = []
        
        for cluster,sequences in self.hashtable.items():
            means = np.mean(sequences,axis=0)
            stds = np.std(sequences,axis=0)
            score_C = np.prod([nbinom.pmf(*convert_params(mean,std),seq)
                               for (mean,std,seq) in zip(means,stds,sequences)])
            self.scores = np.insert(self.scores,score_C)
        self.fitness = np.mean(self.scores)

    def mutate(self):
        '''
        - Select random (non unit) clusters (proportionnaly to their likelihood)
        - Select a random marker for each selected cluster
        - Move it to a new cluster
        '''
        
        inv_fitness = np.max(self.scores)-self.scores
        inv_fitness = [val if len(self.hashtable[cluster]) > 1 else 0
                       for (cluster,val) in enumerate(inv_fitness)]
        switch_probs = inv_fitness/np.sum(inv_fitness)
        switched = np.random.choice(self.n_cluster,
                                    self.mutation_frequency*self.n_cluster,
                                    p=switch_probs)
        markers = []
        for cluster in switched:
            marker = np.random.choice(self.hashtable[cluster])
            markers.append(marker)
            self.assignments[marker] = self.n_cluster
                
        self.setHashTable()

    def setContigencyTable(self):
        tmp = pd.concat([self.assignments]*len(self.assignments),
                        axis=1)
        self.contigency_table = (tmp == tmp.T).astype(int)
        self.contigency_table.columns = self.assignments.index

    def isTogether(self,read,reads_c):
        if self.contigency_table is None:
            self.setContigencyTable()
        return self.contigency_table.loc[read][reads_c].sum()
        
