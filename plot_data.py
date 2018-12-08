import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv("sim1.csv",index_col=0)
clusters = pd.read_csv("solution_clusters.csv",index_col=0)
clusters["marker"] = clusters["marker"].apply(eval)
colors = sns.color_palette("hls",len(clusters.index))

fig,ax = plt.subplots()
for i,(clust,markers) in enumerate(clusters.iterrows()):
    for vec in data.iloc[markers["marker"],:].values:
        ax.plot(vec,color=colors[i])

plt.show()
