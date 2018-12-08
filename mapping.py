#!/usr/bin/env python
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

def score(data):
    if data.shape[0]>1:
        scores = (1-np.percentile(np.corrcoef(data),10,axis=0))/2
    else:
        scores = [ 0.5 ]

    return -np.log(scores)

class Mapping:
    def __init__(self,N,ID=None,assignments={},initialize=True):
        self.id = ID
        self.Nseq = N
        self.assignments = assignments
        self.mutation = { 'freq': 0.25 }
        self.hashtable = None
        self.contingency_table = None
        if initialize:
            self.initialize(N)
        self.assignments['cluster'] = self.assignments['cluster'].astype(int)
        self._n_cluster = None
        self.setHashTable()

    @property
    def n_cluster(self):
        return len(self.assignments['cluster'].unique())

    @classmethod
    def fromCSV(cls,filename='solution.csv'):
        assignments = pd.read_csv(filename,index_col=0)

        mapping = cls(assignments.shape[0],
                      assignments=assignments,
                      initialize=False)

        return mapping

    def initialize(self,N):
        n_cluster = np.random.randint(N/2,N)
        # Make sure every cluster has at least one element
        firsts = np.random.choice(self.Nseq,n_cluster,replace=False)
        others = np.setdiff1d(range(self.Nseq),firsts)

        self.assignments = pd.DataFrame( {i: np.random.randint(0,n_cluster)
                                          for i in others},
                                         index=['cluster']).T
        
        for cluster,first in enumerate(firsts):
            self.assignments.loc[first,"cluster"] = cluster
            
        self.assignments.index.name = 'marker'
        self.assignments['fit'] = 0

    def setHashTable(self):
        self.hashtable = pd.DataFrame(self.assignments['cluster']
                                      .reset_index()
                                      .groupby('cluster')['marker']
                                      .apply(list))

    def setContingencyTable(self):
        self.assignments.sort_index(inplace=True)
        tmp = pd.concat([self.assignments["cluster"]]*self.assignments.shape[0],
                        axis=1)
        tmp.columns = tmp.index
        self.contingency_table = (tmp == tmp.T).astype(int)

    def evaluate(self,sequences):

        scores = (self.hashtable["marker"]
                  .apply(lambda x:sequences.iloc[x].values)
                  .apply(lambda x: score(x))
        )

        scores = pd.concat([self.hashtable["marker"],scores],axis=1)
        scores.columns = ["markers","scores"]
        
        assignments = pd.DataFrame(self.assignments)
        assignments["fit"] = 0

        for cluster,vals in scores.iterrows():
            for i,marker in enumerate(vals['markers']):
                assignments.loc[marker,"fit"] = vals["scores"][i]

        return assignments

    def mutate(self):

        mutation_probs = 1/self.assignments['fit']
        mutation_probs /= mutation_probs.sum()

        n_mutations = max(1,int(self.mutation['freq']*self.assignments.shape[0]))
        mutated_markers = np.random.choice(self.assignments.index,
                                           n_mutations,
                                           p=mutation_probs,
                                           replace=False)

        new_assignments = [np.random.choice(1+self.n_cluster)
                           for _ in mutated_markers]
        
        for marker,assignment in zip(mutated_markers,new_assignments):
            self.assignments.loc[marker,"cluster"] = assignment
            
        self.setHashTable()

    def plot(self,data):
        plot_data = data.reset_index()
        plot_data['cluster'] = self.assignments['cluster'].values
        plot_data = pd.melt(plot_data,id_vars=['cluster','index'])

        g = sns.FacetGrid(plot_data,col='cluster',col_wrap=3,sharey=False)
        g.map(sns.lineplot,'variable','value','index',estimator=None)
        plt.show()
