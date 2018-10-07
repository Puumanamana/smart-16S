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

    def __init__(self,ID,N):
        self.id = ID
        self.hashtable = {}
        self.scores = np.array([])
        self.mutation_frequency = 0.1
        self.initialize(N)

    def initialize(self,N):
        self.n_cluster = np.random.randint(1,N/4)
        assignments = pd.DataFrame( [(i,np.random.randint(1,self.n_cluster)) for i in range(N)],
                                    columns = ['marker','cluster'])
        self.hashtable = assignments.groupby('cluster')['marker'].apply(list)
        
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
        inv_fitness = np.max(self.scores)-self.scores
        switch_probs = inv_fitness/np.sum(inv_fitness)
        switched = np.random.choice(self.n_cluster,
                                    self.mutation_frequency*self.n_cluster,
                                    p=switch_probs)
        markers = []
        for cluster in switched:
            marker = np.random.choice(self.hashtable[cluster])
            markers.append(marker)
            self.hashtable[cluster].remove(marker)
        self.hashtable.append({self.n_cluster: markers}, ignore_index=True)

        for cluster in self.hashtable.index:
            if len(self.hashtable.loc[cluster]) ==0:
                self.hashtable.drop(cluster,inplace=True)
        
