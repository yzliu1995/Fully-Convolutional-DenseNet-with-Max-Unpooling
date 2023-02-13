"""
This file is developed by Joe Liu
"""

import os
import math
import numpy as np 
import time

# Imports for plotting
import matplotlib.pyplot as plt
import optparse
from matplotlib.colors import to_rgba
from mycolorpy import colorlist as mcp
import seaborn as sns
import matplotlib as mpl
import random
mpl.rcParams["legend.markerscale"] = 0.5
import pandas as pd

parser = optparse.OptionParser()

parser.add_option("-l", "--file",
                  action="store", type="string", dest="filename1", help = "one input file for FCDenseNet")

parser.add_option("-a", "--anotherFile",
                  action="store", type="string", dest="filename2", help = "the other input file for FCDenseNet V2")

parser.add_option("-i", "--title",
                  action="store", type="string", dest="title", help = "title for the figure")

parser.add_option("-n", "--name",
                  action="store", type="string", dest="name", help = "filename for the figure")


(options, args) = parser.parse_args()

df_dense_1 = pd.read_csv(options.filename1)
df_dense_1 = df_dense_1.round(4)
df_dense_1["Model"] = "FCDenseNet"
df_v2_1 = pd.read_csv(options.filename2)
df_v2_1 = df_v2_1.round(4)
df_v2_1["Model"] = "FCDenseNet V2"

df_c_1 = pd.concat([df_v2_1, df_dense_1])
df_c_1 = df_c_1.melt(id_vars="Model")

g2 = sns.boxplot(x = "variable", y = "value", hue = "Model", data = df_c_1)
g2.set_xticks(range(7))
g2.set_xticklabels(["Sensitivity", "Specificity", "Dice coefficient", "g-mean", "IoU", "Pixel Accuracy"], rotation=15)
g2.set_xlabel("")
g2.set_ylabel("")
plt.title(options.title)
plt.savefig("./figures/"+options.name+".pdf", dpi = 600)

dense_stat = ["{:.4f} ({:.4f})".format(f[0], f[1]) for idx, f in enumerate(zip(df_dense_1.mean(), df_dense_1.std()))]
v2_stat = ["{:.4f} ({:.4f})".format(f[0], f[1]) for idx, f in enumerate(zip(df_v2_1.mean(), df_v2_1.std()))]

statD = {}
for i in range(len(df_dense_1.columns)-1):
  statD[df_dense_1.columns[i]] = []
  statD[df_dense_1.columns[i]].append(dense_stat[i])
  statD[df_dense_1.columns[i]].append(v2_stat[i])

statDF = pd.DataFrame(statD, index = ["FCDenseNet", "FCDenseNet V2"])

statDF.to_csv("./results/"+options.name+".csv")
