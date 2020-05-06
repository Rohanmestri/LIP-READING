from keras import utils
from keras.models import Sequential,model_from_yaml
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LSTM, GRU
from keras.layers import TimeDistributed, Conv3D,MaxPooling3D, ZeroPadding3D
from keras import backend as K
from imutils import face_utils

import imutils
import dlib
import numpy as np
import cv2
import math
import random
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
#########################################################################################33########
predictor_model = "shape_predictor_68_face_landmarks.dat"

face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
################################### Preprocessing videos by applying haar cascades to isolate the face ################################################################


def preprocessing(cap):
 
 euclid_dist = np.empty(shape=(25,20))
 terminate_flag,count,inner_count = 0,0,0

 #randomizing the 25 frames.
 rand_list = [0,1,2,3,25,26,27,28]
 key = random.choice(rand_list)
 if key>24:
     no_frame = list(range(key - 25,key))
 else:
     no_frame = list(range(key,key+25))
     
 print(no_frame)
 while(cap.isOpened()):
   ret, frame = cap.read()
   if ret == True:
    if count in no_frame:
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     detected_faces = face_detector(gray, 1)
     for i, face_rect in enumerate(detected_faces):
      pose_landmarks = face_pose_predictor(gray, face_rect)
      pose_landmarks = face_utils.shape_to_np(pose_landmarks)
      mean_x,mean_y = 0,0
      for (x, y) in pose_landmarks[48:68]:
        mean_x = mean_x + x
        mean_y = mean_y + y
      mean_x = mean_x/20
      mean_y = mean_y/20
      int_count = 0
      for (x, y) in pose_landmarks[48:68]:
        euclid_dist[inner_count,int_count] = math.sqrt(math.pow((mean_x-x),2)+math.pow((mean_y-y),2))
        int_count += 1
     inner_count +=1  
    count+=1
   else:
     break
   
 cap.release()
 cv2.destroyAllWindows()

 return euclid_dist,1

stride = 1
frame_list = [None] * 25


def RealTime_preprocess(path,length):
   
   for i in range(0,(length-25),stride):
       for j in range(25):
           current_frame = i+j
           frame_list[j] = current_frame

       euclid_dist = np.empty(shape=(25,20))
       cap = cv2.VideoCapture(path)
       
       count,inner_count = 0,0
       while(cap.isOpened()):
         ret, frame = cap.read()
         #frame = cv2.transpose(frame)
         #frame = cv2.flip(frame,0)
         if ret == True:
           if count in frame_list:
              gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
              detected_faces = face_detector(gray, 1)
              for i, face_rect in enumerate(detected_faces):
                 pose_landmarks = face_pose_predictor(gray, face_rect)
                 pose_landmarks = face_utils.shape_to_np(pose_landmarks)
                 mean_x,mean_y = 0,0
                 for (x, y) in pose_landmarks[48:68]:
                   mean_x = mean_x + x
                   mean_y = mean_y + y
                 mean_x = mean_x/20
                 mean_y = mean_y/20
                 int_count = 0
                 for (x, y) in pose_landmarks[48:68]:
                     euclid_dist[inner_count,int_count] = math.sqrt(math.pow((mean_x-x),2)+math.pow((mean_y-y),2))
                     int_count += 1       
              inner_count +=1  
           count+=1
         else:
            break
       
       euclid_dist = euclid_dist.reshape((1,25,20))
       euclid_dist = euclid_dist/4.16
       
       y = int(loaded_model.predict_classes(euclid_dist))
       y2 = loaded_model.predict_proba(euclid_dist)
       temp = list(words.keys())
       cap.release()
       
   
           


################################## Preparing Data for a single word ####################################################################################################

def word_to_data(string,n_train,n_val,n_test,dict_words):
  print("Preparing the data for the word %s" %string)
 
  #_____TRAINING SET___# 
  first_flag,actual_count = 0,0
  for vid in range(n_train):
   if vid<9:
    cap = cv2.VideoCapture('C:/Users/Rohan/Videos/lipread_mp4/{0}/train/{1}_0000{2}.mp4'.format(string,string,str(vid+1)))
   elif vid>=9 and vid<99:
    cap = cv2.VideoCapture('C:/Users/Rohan/Videos/lipread_mp4/{0}/train/{1}_000{2}.mp4'.format(string,string,str(vid+1)))
   elif vid>=99 and vid<999:
    cap = cv2.VideoCapture('C:/Users/Rohan/Videos/lipread_mp4/{0}/train/{1}_00{2}.mp4'.format(string,string,str(vid+1)))
   elif vid==999:
    cap = cv2.VideoCapture('C:/Users/Rohan/Videos/lipread_mp4/{0}/train/{1}_01000.mp4'.format(string,string))
   
   temp,bool_flag = preprocessing(cap)  
   if bool_flag == 1 and first_flag == 0:
    X_train = temp
    first_flag = 1
    actual_count += 1
    
   elif bool_flag == 1:
    X_train = np.append(X_train,temp,axis=0)
    actual_count += 1
   print(actual_count)
  X_train = X_train.reshape(actual_count,25,20).astype('float32')
 
  y_train = [None]*actual_count 
  for i in range(actual_count):
    y_train[i] = dict_words[string]
  

  '''
  #_____VALIDATION SET_____#
  first_flag,actual_count = 0,0
  for vid in range(n_val):
   if vid<9:
    cap = cv2.VideoCapture('C:/Users/Rohan/Videos/lipread_mp4/{0}/val/{1}_0000{2}.mp4'.format(string,string,str(vid+1)))
   elif vid>=9 and vid<50:
    cap = cv2.VideoCapture('C:/Users/Rohan/Videos/lipread_mp4/{0}/val/{1}_000{2}.mp4'.format(string,string,str(vid+1)))

   temp,bool_flag = preprocessing(cap)  
   
   if bool_flag == 1 and first_flag == 0:
    X_val = temp
    first_flag = 1
    actual_count += 1
    
   elif bool_flag == 1:
    X_val = np.append(X_val,temp,axis=0)
    actual_count += 1

  X_val = X_val.reshape(actual_count,25,20).astype('float32')
 
  y_val = [None]*actual_count 
  for i in range(actual_count):
    y_val[i] = dict_words[string]'
  '''

    #_____VALIDATION SET_____#
  first_flag,actual_count = 0,0
  for vid in range(n_val):
   cap = cv2.VideoCapture('C:/Users/Rohan/Videos/lipread_mp4/{0}/train/{1}_00{2}.mp4'.format(string,string,str(vid+1+750)))
    
   temp,bool_flag = preprocessing(cap)  
   
   if bool_flag == 1 and first_flag == 0:
    X_val = temp
    first_flag = 1
    actual_count += 1
    
   elif bool_flag == 1:
    X_val = np.append(X_val,temp,axis=0)
    actual_count += 1

  X_val = X_val.reshape(actual_count,25,20).astype('float32')
 
  y_val = [None]*actual_count 
  for i in range(actual_count):
    y_val[i] = dict_words[string]

    
   
  #_____TEST SET_____#
  first_flag,actual_count = 0,0
  for vid in range(n_test):
   if vid<9:
    cap = cv2.VideoCapture('C:/Users/Rohan/Videos/lipread_mp4/{0}/test/{1}_0000{2}.mp4'.format(string,string,str(vid+1)))
   elif vid>=9 and vid<50:
    cap = cv2.VideoCapture('C:/Users/Rohan/Videos/lipread_mp4/{0}/test/{1}_000{2}.mp4'.format(string,string,str(vid+1)))

   temp,bool_flag = preprocessing(cap)  
   
   if bool_flag == 1 and first_flag == 0:
    X_test = temp
    first_flag = 1
    actual_count += 1
    
   elif bool_flag == 1:
    X_test = np.append(X_test,temp,axis=0)
    actual_count += 1

  X_test = X_test.reshape(actual_count,25,20).astype('float32')
 
  y_test = [None]*actual_count 
  for i in range(actual_count):
    y_test[i] = dict_words[string]

  y_train = np.asarray(y_train)
  y_test = np.asarray(y_test)
  y_val = np.asarray(y_val)
  
  return X_train,y_train,X_val,y_val,X_test,y_test


################################### Concatenating the individual words to create a global numpy dataset ###################################################################

def create_dataset(list_of_words):
 first_flag = 0
 for word in list_of_words.keys():
   trainX,trainY,valX,valY,testX,testY = word_to_data(word,350,50,50,list_of_words)

   if first_flag == 0:
     X_train = trainX
     X_test = testX
     X_val = valX
     y_train = trainY
     y_test = testY
     y_val = valY
     first_flag = 1
   else:
     X_train = np.append(X_train,trainX,axis=0)
     X_test = np.append(X_test,testX,axis=0)
     X_val = np.append(X_val,valX,axis=0)
     y_train = np.append(y_train,trainY,axis=0)
     y_test = np.append(y_test,testY,axis=0)
     y_val = np.append(y_val,valY,axis=0)

 y_train = utils.to_categorical(y_train)
 y_test = utils.to_categorical(y_test)
 y_val = utils.to_categorical(y_val)
 
 num_classes = y_train.shape[1]

 return X_train,X_test,X_val,y_val,y_train,y_test,num_classes


######################################### Loading the prepared data ####################################################################################################

words = {'ABUSE':1,'BLACK':2,'EXACTLY':3,'CRIME':4}
#Uncomment to save new data
'''
X_train,X_test,X_val,y_val,y_train,y_test,num_classes = create_dataset(words) 

np.save('X_train3.npy',X_train)
np.save('X_test3.npy',X_test)
np.save('X_val3.npy',X_val)
np.save('y_train3.npy',y_train)
np.save('y_test3.npy',y_test)
np.save('y_val3.npy',y_val)
'''

###################################### Exporting inference graphs #######################################################################################################

MODEL_NAME = 'lip_reading'

def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out2', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")

################################# Creating the Neural Network Model#####################################################################################################

X_train = np.load('X_train3.npy')/2.56
X_test = np.load('X_test3.npy')/2.56
X_val = np.load('X_val3.npy')/2.56
y_train = np.load('y_train3.npy')
y_test = np.load('y_test3.npy')
y_val = np.load('y_val3.npy')
num_classes = y_train.shape[1]


y_train = np.delete(y_train,0,1)
y_val = np.delete(y_val,0,1)
y_test = np.delete(y_test,0,1)



#creating the network
model = Sequential()
model.add(LSTM(512))
model.add(Dropout(0.30))
model.add(Dense(4, activation='softmax'))

#Fitting the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, y_train,validation_data=[X_val,y_val], epochs=120, batch_size=256)
model.summary()

#Saving the model - to be loaded in the app
export_model(tf.train.Saver(), model, ["gru_1_input"], "dense_1/Softmax")


#Testing the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
scores,acc = model.evaluate(X_test, y_test,verbose=0)
model.summary()
print("Accuracy on the test set: %s"%str(acc))

############################# Test Vector #####################################
'''
path = 'C:/Users/Rohan/Videos/lipread_mp4/ABSOLUTELY/test/ABSOLUTELY_00030.mp4'
#path = 'C:/Users/Rohan/Documents/hog/ABOUT_00002.mp4'
cap = cv2.VideoCapture(path)
test,bool_flag = preprocessing(cap)
test = test.reshape((1,25,20))
y = int(loaded_model.predict_classes(test))
y2 = loaded_model.predict_proba(test)
temp = list(words.keys())

cap = cv2.VideoCapture(path)
while(cap.isOpened()):
 ret, frame = cap.read()
 if ret == True:
  cv2.imshow("test_video",frame)
  cv2.waitKey(80)
 else:
   break
cap.release()
cv2.destroyAllWindows()

print("The predicted word is %s" %(temp[y]))
#print(y2)
'''

############################ Indian Accent #######################################

'''
#path = 'C:/Users/Rohan/Videos/lipread_mp4/BLACK/test/BLACK_00030.mp4'
path = 'C:/Users/Rohan/Documents/hog/indian_accent_abuse.mp4'
cap_temp = cv2.VideoCapture(path)
length = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
RealTime_preprocess(path,length)
'''



