# Lip Reading using Deep Learning

A real-time android application which captures video frames from a person speaking a particular word and runs the trained inference model in the backend to predict the spoken word. 

## Implementation

* The Feature Extraction Stage is carried out by a HOG-SVM classifier pretrained to detect 68 landmark points on a given face. 
* The features surrounding the lip-region are stored in the form of Euclidean Distances. 
* These 20 features from each frame (29 frames - with 2 frames clipped from the start and the end) are fed into LSTM layers.
* The output of the final LSTM layer is attached to FC layers, which give the probabilistic output corresponding to each word.

## Tech Stack 

*Prototype - Python, Keras, Tensorflow, Opencv, Numpy, Dlib 
