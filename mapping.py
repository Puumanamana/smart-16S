#!/usr/bin/env python
import numpy as np
from scipy.stats import nbinom
import pandas as pd

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from generate_data import convert_params

ALPHA = 5 # estimate alpha from the data?

def logNB1m(sequences,thresholds=[1e-3,1e-1]):
    scores = []
    for seq in sequences.T:
        seq_pos = seq[seq>0]
        if len(seq_pos) > 0:
            mu = np.floor(np.median(seq_pos))
            params = convert_params(mu,mu/ALPHA)
            
            score_seq = nbinom.pmf(seq,*params) / nbinom.pmf(mu,*params)
            score_seq = [min(max(x,thresholds[0]),1-thresholds[1]) for x in score_seq]
            scores.append(score_seq)
    return -np.log(1-np.array(scores)) # returns -{log(1-P[v])}_{marker,sample}

class Mapping:
    def __init__(self,N,ID=None,assignments={},initialize=True):
        self.id = ID
        self.Nseq = N
        self.assignments = assignments
        self.mutation = { 'freq': 0.2 }
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
                  .apply(logNB1m)
                  .apply(lambda x: np.percentile(x,25,axis=0))#np.exp(np.log(x).mean(axis=0)))
        )

        # Scores = {cluster: [marker_scores across samples] }
        scores = pd.concat([self.hashtable["marker"],scores],axis=1)
        scores.columns = ["markers","scores"]
        
        assignments = pd.DataFrame(self.assignments)
        assignments["fit"] = 0

        for cluster,vals in scores.iterrows():
            for i,marker in enumerate(vals['markers']):
                assignments.loc[marker,"fit"] = vals["scores"][i]
                
        return assignments

    def mutate(self):
        # probs_per_cluster = self. / self.scores.sum()
        # mutation_probs = self.assignments.reset_index() 
        # mutation_probs['probs'] = probs_per_cluster[mutation_probs.cluster].values

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
            
        # clusters = self.hashtable.index.tolist()

        # def choose(entry):
        #     new_cluster = np.setdiff1d(range(max(clusters)+2),clusters)[0]
        #     p_else = (1-entry[2])/self.n_cluster
        #     probs = [ entry[2] if c==entry[1] else p_else
        #               for c in clusters+[new_cluster] ]
        #     return np.random.choice(clusters+[new_cluster],p=probs)
        
        # self.assignments = mutation_probs.apply(choose,axis=1)
        # self.assignments.index.name = 'marker'
        # self.assignments.name = 'cluster'

        self.setHashTable()

    def plot(self,data):
        plot_data = data.reset_index()
        plot_data['cluster'] = self.assignments['cluster'].values
        plot_data = pd.melt(plot_data,id_vars=['cluster','index'])

        g = sns.FacetGrid(plot_data,col='cluster',col_wrap=3,sharey=False)
        g.map(sns.lineplot,'variable','value','index',estimator=None)
        # sns.lineplot(x='variable',
        #              y='value',
        #              hue='cluster',
        #              units='index',
        #              estimator=None,
        #              lw=1,
        #              palette=sns.color_palette("hls",self.n_cluster),
        #              data=plot_data)
        plt.show()

    # def isTogether(self,read,reads_c):
    #     if self.contingency_table is None:
    #         self.setContingencyTable()
    #     return self.contingency_table.loc[read][reads_c].sum()
