# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 10:59:12 2019

@author: majit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import PCA


def show_orignal_images(pixels):
	#Displaying Orignal Images
	fig, axes = plt.subplots(6, 10, figsize=(11, 7),
	                         subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(np.array(pixels)[i].reshape(120, 120), cmap='gray')
	plt.show()


def show_eigenfaces(pca):
	#Displaying Eigenfaces
	fig, axes = plt.subplots(3, 8, figsize=(9, 4),
	                         subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(pca.components_[i].reshape(120, 120), cmap='gray')
	    ax.set_title("PC " + str(i+1))
	plt.show()
    
    
df = pd.read_csv("faces_data_train.csv")
#df = df.drop([1]) #just for this data
targets = df["target"]
pixels = df.drop(["target"],axis=1)

show_orignal_images(pixels)


pca = PCA(n_components=5).fit(pixels)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()

show_eigenfaces(pca)
