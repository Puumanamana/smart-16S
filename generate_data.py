#!/usr/bin/env python

import re
from scipy.stats import nbinom
import numpy as np
import pandas as pd

def is_present(bern_prob):
    return True if np.random.rand() < bern_prob else False


def convert_params(mu, theta):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    if mu == 0:
        mu = 1
    r = theta
    var = mu + 1 / r * mu ** 2
    if var == 0:
        var = 0.5
    p = (var - mu) / var
    return r, 1 - p


def main():
    # G for genome and E for ESV
    nb_samples = 50

    markers = {0: [0],
               1: [1,2],
               2: [3,4],
               3: [5],
               4: [6,7],
               5: [8],
               6: [9,10,11,12,13],
               7: [14,15,16,17],
               8: [18],
               9: [19], 
               10: [20,21],
               11: [22],
               12: [23,24],
               13: [25]
    }

    markers = pd.Series(markers,name="marker")
    genomes = markers.index

    # markers_df = markers_df.apply(lambda x: list(map(lambda y: int(y.replace('M','')),x)))
    # markers_df.index = range(markers_df.shape[0])
    markers.index.name = "cluster"

    markers.to_csv("solution_clusters.csv",header=True)

    assignments = {}
    n_markers = markers.apply(len).sum()

    for marker in range(n_markers):
        assignments[marker] = [genome for genome in genomes
                               if marker in markers[genome]][0]

    assignments_df = pd.Series(assignments,name="cluster")
    assignments_df.index.name = 'marker'
    assignments_df.to_csv("solution.csv",header=True)

    correlations = {} # {"G4": ("G3" , 3.5) , "G12": ("G11", 2.7)}

    # proba of observance genome in sample
    # should be randomly sampled for now static
    prob_g_present = 0.96
    # high_abundance = set(["G7"])
    # low_abundance = set(["G8"])

    data = {}
    markers_present = {}

    for sample_nb in range(nb_samples):
        sample = 'S{}'.format(sample_nb)
        genomes_present = {}

        high_abundance = markers.sample(frac=.2).index
        low_abundance = markers.sample(frac=.25).index

        for genome in genomes:
            # if  genome is correlated to an anchor (reference) genome
            # if genome in correlations and correlations[genome][0] in genomes_present:
            #     anchor = correlations[genome][0]
            #     anchor_corr_coefficient = correlations[genome][1]
            #     anchor_corr_noise = np.random.normal(0,2)
            #     genomes_present[genome] = int (genomes_present[anchor] * anchor_corr_coefficient + anchor_corr_noise)

            if is_present(prob_g_present):
                if genome in high_abundance:
                    high_abundance_p, high_abundance_q = convert_params(300, 1000)
                    genomes_present[genome] = nbinom.rvs(high_abundance_p, high_abundance_q)
                elif genome in low_abundance:
                    low_abundance_p, low_abundance_q = convert_params(3, 1000)
                    genomes_present[genome] = nbinom.rvs(low_abundance_p, low_abundance_q)
                else:
                    p, q = convert_params(100, 100)
                    genomes_present[genome] = nbinom.rvs(p, q)
        data[sample] = genomes_present

        markers_present[sample] = {}

        for genome, abundance in genomes_present.items():

            p, q = convert_params(abundance, 1000)
            for marker in markers.loc[genome]:
                marker_abundance = nbinom.rvs(p, q)
                if marker not in markers_present[sample]:
                    markers_present[sample][marker] = marker_abundance
                else:
                    markers_present[sample][marker] += marker_abundance             
                # print(sample,genome,abundance,marker,marker_abundance,p,q)

    data = pd.DataFrame(markers_present)
    sort_fn = lambda x: int(re.findall('\d+',x)[0])
    data = data.reindex(columns=sorted(data.columns,key=sort_fn),
                        index=sorted(data.index)).fillna(0)
    data.astype(int).to_csv('sim1.csv')

if __name__ == '__main__':
    main()
