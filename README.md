# Lip Reading using Deep Learning

A Real-Time Android Application which captures video frames from a person speaking a particular word and runs the trained inference model in the backend to predict the spoken word. 

## Implementation

* The Feature Extraction Stage is carried out by a HOG-SVM classifier pretrained to detect 68 landmark points on a given face. 
* The features surrounding the lip-region are stored in the form of Euclidean Distances. 
* These 20 features from each frame (29 frames - with 2 frames clipped from the start and the end) are fed into LSTM layers.
* The output of the final LSTM layer is attached to FC layers, which give the probabilistic output corresponding to each word.

## Tech Stack 

* **Dataset** - Lip Reading in the Wild (BBC) 
* **Prototype** - Python, Keras, Tensorflow, Opencv, Numpy, Dlib 
* **Android App** - Java, Tensorflow Lite (Android), OpenCV-Android, Dlib-Android(Tzutalin)

## How to Use

* Clone the repository.
* Have acces to the dataset and make sure to place it in the appropriate directory.
* First run the generate_model.py found in the Python_Prototype directory to generate the model using the dataset.
* Export the saved model and incorporate it in the source code of the Android Application.
* Make necessary changes in the Camera Interface source codes according to the mobile phone's specifications.
* Run the application in Android for the words on which the model is trained.
