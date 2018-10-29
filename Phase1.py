import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.getcwd()

data = pd.read_csv('//prism.nas.gatech.edu/ssidhwa3/vlab/downloads/gt.txt', sep=";", header=None)
data.columns = ["img", "x1", "y1", "x2", "y2", "id"]


print("Given Training data has image files with traffic signs numbers as :", len(data['img'].unique()))
print("Given Training data has traffic signs class numbers as :", len(data['id'].unique()))
print("Given Training data has traffic signs instance numbers as :", data['id'].count())
#pd.value_counts(data['id'], sort=False).plot.bar()
print(pd.value_counts(data['id'], sort=True))
plt.hist(data['id'], bins=82)

plt.show()
