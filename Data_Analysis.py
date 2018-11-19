import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import os
os.getcwd()

# TrainIJCNN2013    '../data/TrainIJCNN2013/gt.txt'
#data = pd.read_csv('../data/TrainIJCNN2013/gt.txt', sep=";", header=None)

# FullIJCNN2013     '../data/FullIJCNN2013/gt.txt'
data = pd.read_csv('C:/Users/shubham sidhwa/Downloads/TrainIJCNN2013/TrainIJCNN2013/gt.txt', sep=";", header=None)
data.columns = ["img", "x1", "y1", "x2", "y2", "id"]
#data=data[:3]

print("Train has image files with traffic signs numbers:", len(data['img'].unique()))
print("Train has traffic signs class numbers:", len(data['id'].unique()))
print("Train has traffic signs instance numbers:", data['id'].count())
#pd.value_counts(data['id'], sort=False).plot.bar()
print(pd.value_counts(data['id'], sort=True))
data['id'].hist(bins=86)

plt.show()
