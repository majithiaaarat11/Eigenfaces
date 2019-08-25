# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
import pandas as pd

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


#TODO: Better system for ids
ids = {'bhuvaneshwar':1,'jasprit':2,'virat':3}
 
 
def get_images_uniform_and_labels2(path):
    
    labels = []
    image_paths = []
    images = []
    
    ls = os.listdir(path)
    for l in ls:
        iss = os.listdir(os.path.join(path,l))
        for i in iss:
            image_paths.append(os.path.join(path,l,i))

    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        
        nbr = image_path.split('\\')[1]
        
        faces = faceCascade.detectMultiScale(image)
        print(image_path)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(cv2.resize(image[y: y + h, x: x + w],(120,120)).reshape((1,120*120)))
            labels.append(ids[nbr])

    #list to np arrays
    #TODO: Normalise    
    X = np.concatenate(images)
    df= pd.DataFrame(X)
    df['target'] = pd.Series(labels, index=df.index)
    
    return X, df

X_train, df_train = get_images_uniform_and_labels2('train')
np.save('X_train',X_train)
df_train.to_csv("faces_data_train.csv", index = False)

_, df_test = get_images_uniform_and_labels2('test')
df_test.to_csv("faces_data_test.csv", index = False)



