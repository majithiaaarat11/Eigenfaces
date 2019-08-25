# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 10:59:12 2019

@author: majit
"""

# reference: https://github.com/spmallick/learnopencv/blob/master/EigenFace/EigenFace.py

import numpy as np
impo1rt pandas as pd
import cv2

	#Displaying Orignal Images
# Add the weighted eigen faces to the mean face 
def createNewFace(*args):
	# Start with the mean image
	output = averageFace
	
	# Add the eigen faces with the weights
	for i in range(0, NUM_EIGEN_FACES):
		'''
		OpenCV does not allow slider values to be negative. 
		So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
		''' 
		sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
		weight = sliderValues[i] - MAX_SLIDER_VALUE/2
		output = np.add(output, eigenFaces[i] * weight)

	# Display Result at 2x size
	output = cv2.resize(output, (0,0), fx=2, fy=2)
	cv2.imshow("Result", output)

def resetSliderValues(*args):
	for i in range(0, NUM_EIGEN_FACES):
		cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2));
	createNewFace()

if __name__ == '__main__':

	# Number of EigenFaces
	NUM_EIGEN_FACES = 6

	# Maximum weight
	MAX_SLIDER_VALUE = 255

	# Size of images
	sz = (120,120)

	#TODO: Same thing with RGB channels

	# Create data matrix for PCA.
	data = np.load('X.npy')
	data = np.delete(data,1,0)
	data = data/255

	# Compute the eigenvectors from the stack of images created
	print("Calculating PCA ", end="...")
	mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
	print ("DONE")

	averageFace = mean.reshape(sz)

	eigenFaces = []; 

	for eigenVector in eigenVectors:
		eigenFace = eigenVector.reshape(sz)
		eigenFaces.append(eigenFace)

	# Create window for displaying Mean Face
	cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
	
	# Display result at 2x size
	output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
	cv2.imshow("Result", output)

	# Create Window for trackbars
	cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

	sliderValues = []
	
	# Create Trackbars
	for i in range(0, NUM_EIGEN_FACES):
		sliderValues.append(int(MAX_SLIDER_VALUE/2))
		cv2.createTrackbar( "Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2), MAX_SLIDER_VALUE, createNewFace)
	
	# You can reset the sliders by clicking on the mean image.
	cv2.setMouseCallback("Result", resetSliderValues);
	
	print('''Usage:
	Change the weights using the sliders
	Click on the result window to reset sliders
	Hit ESC to terminate program.''')

	cv2.waitKey(0)
	cv2.destroyAllWindows()