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

    def __init__(self,N,ID=None,assignments={}):
        self.id = ID
        self.Nseq = N
        self.assignments = assignments
        self.scores = np.array([])
        self.mutation_frequency = 0.1
        self.hashtable = None
        self.contigency_table = None
        self.initialize(N)

    def initialize(self,N):
        self.n_cluster = np.random.randint(2,N/4)
        # Make sure every cluster has at least one element
        firsts = np.random.choice(self.Nseq,self.n_cluster,replace=False)
        others = np.setdiff1d(range(self.Nseq),firsts)
        
        self.assignments = pd.Series( {i: np.random.randint(0,self.n_cluster)
                                       for i in others},
                                      name='cluster')
        for cluster,first in enumerate(firsts):
            self.assignments[first] = cluster
        self.assignments.index.name = 'marker'
        self.setHashTable()

    def setHashTable(self):
        self.hashtable = (self.assignments
                          .reset_index()
                          .groupby('cluster')['marker']
                          .apply(list))
        clusterId_absent = np.setdiff1d(range(self.n_cluster),self.hashtable.index)
        empty_df = pd.Series([[]]*len(clusterId_absent),
                             index=clusterId_absent)
        self.hashtable = pd.concat([self.hashtable,empty_df])

    def setContigencyTable(self):
        tmp = pd.concat([self.assignments]*len(self.assignments),
                        axis=1)
        tmp.columns = tmp.index
        self.contigency_table = (tmp == tmp.T).astype(int)
        
    def evaluate(self,sequences):
        self.scores = []
        
        for cluster,idx_seqC in self.hashtable.items():
            seqC = sequences.iloc[idx_seqC].T
            params = seqC.apply([np.mean,np.std]).T
            
            log_likelihood_C = np.mean([nbinom.logpmf(*convert_params(mean,std),seq[1])
                                        for (mean,std,seq) in
                                        zip(params['mean'],params['std'],seqC.iterrows())])
            if np.isnan(log_likelihood_C):
                log_likelihood_C = -1e5
            self.scores = np.append(self.scores,log_likelihood_C)
        self.fitness = np.median(self.scores)

    def mutate(self):
        '''
        - Select random (non unit) clusters (proportionnaly to their likelihood)
        - Select a random marker for each selected cluster
        - Move it to a new cluster
        '''
        
        inv_fitness = np.max(self.scores)-self.scores+0.001
        inv_fitness = [val if len(self.hashtable[cluster]) > 1 else 0
                       for (cluster,val) in enumerate(inv_fitness)]
        switch_probs = inv_fitness/np.sum(inv_fitness)
        switched = np.random.choice(self.n_cluster,
                                    int(self.mutation_frequency*self.n_cluster),
                                    p=switch_probs)
        markers = []
        for cluster in switched:
            marker = np.random.choice(self.hashtable[cluster])
            markers.append(marker)
            self.assignments[marker] = self.n_cluster
                
        self.setHashTable()


    def isTogether(self,read,reads_c):
        if self.contigency_table is None:
            self.setContigencyTable()
        return self.contigency_table.loc[read][reads_c].sum()
        
