# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:47:00 2019

@author: majit
"""

import cv2, os
import numpy as np
import pandas as pd

shape = (64,64)
path = "data2"

def get_image_paths(path):
    
    image_paths = []
    
    ls = os.listdir(path)
    for l in ls:
        iss = os.listdir(os.path.join(path,l))
        for i in iss:
            image_paths.append(os.path.join(path,l,i))

    return image_paths



image_paths = get_image_paths(path)


def get_images_and_labels(image_paths):
    
    images = []
    target_names = []
    
    for image_path in image_paths:
        
        im = cv2.imread(image_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        images.append(cv2.resize(gray,shape).reshape((1,shape[0]*shape[1])))
        target_names.append(image_path.split('\\')[1][:-17])
        
        
    X = np.concatenate(images)
    df= pd.DataFrame(X)
    df['target_names'] = pd.Series(target_names, index=df.index)
    
    df['target'] = pd.factorize(df['target_names'])[0]
    
    return X, df


X, df = get_images_and_labels(image_paths)


np.save('X_tyasa',X)
df.to_csv("faces_data_tyasa.csv", index = False)



        