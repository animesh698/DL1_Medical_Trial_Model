import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_samples=[]
train_labels=[]

#Generating samples and labels using randint funtion
#We append 1 if older and 0 if younger
for i in range(50):
	#5% of younger individuals who did experience side effects
	random_younger=randint(13,64)
	train_samples.append(random_younger)
	train_labels.append(0)
    
    #5% of older individuals who did experience side effects
	random_older=randint(65,100)
	train_samples.append(random_older)
	train_labels.append(1)


for i in range(1000):
	#95% of younger individuals who did experience side effects
	random_younger=randint(13,64)
	train_samples.append(random_younger)
	train_labels.append(0)

	#95% of older indivisuals who did experience side effects
	random_older=randint(65,100)
	train_samples.append(random_older)
	train_labels.append(1)	
# If you print train_samples and train_labels as it is it will display results in a list,since they are declared as a list
# Hence we use a for loop to display individual samples or labels

# for i in train_labels:
# 	print(i)

#keras requiers samples and labels to be in a numpy array or a list of numpy array
train_labels=np.array(train_labels)
train_samples=np.array(train_samples)

#however the net won't be able to learn from this training data the way it is(13,100)
#Hence we make use of scikit-learn' MinMaxScaler which will scale down from (13,100)  to the rang especified(0,1)

scaler=MinMaxScaler(feature_range=(0,1))
scaled_train_samples=scaler.fit_transform((train_samples).reshape(-1,1)) #reshape function is a technical formality,of working with a 1d array

# for i in scaled_train_samples:
# 	print(i)

#keras Sequential class is a linear stack of array.
#model accepts an array.
#each of which is the layer
#model=Sequential([l1,l2..]),this is how you pass the layers in the constructor
#or model.add(l1,l2..)
#similarly the activation funtion can be passed as a parameter or another layer as model.add(activation='relu')

#Only the first layer ie the 1st dense layer reqiires the input_shape
model=Sequential([
	Dense(16,input_shape=(1,),activation='relu'), 
	Dense(32,activation='relu'),
	Dense(2,activation='softmax')
	])
# print (model.summary())

#Adam is an optimizer funtion,metrics to check the performance of the model
model.compile(Adam(lr=.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Training
#Epochs: No of traning runs; shuffle:every time model goes through data, it shuffles 
#the data or order in which it runs through the data,by default it remains True for the 
#fit function.
#verbose:
model.fit(scaled_train_samples,train_labels,validation_split=0.1,batch_size=10,epochs=20,shuffle=True,verbose =2)

#Validation to avoid overfitting
#struct is an array that contains a tuple of a sample and the label
#valid_set=[(sample,label),(sample,label),...,(sample,label)]
#include this is above model.fit using validaion_data
# or validation_split=0-1
#If validation_split is mentioned then shuffle becomes false hence in every epoch the validation data remains the same i.e last 10% in every run and does not shuffle
test_samples=[]
test_labels=[]

for i in range(10):
	random_younger=randint(13,64)
	test_samples.append(random_younger)
	test_labels.append(1)

	random_older=randint(65,100)
	test_samples.append(random_older)
	test_labels.append(0)

for i in range(200):
	random_younger=randint(13,64)
	test_samples.append(random_younger)
	test_labels.append(0)

	random_older=randint(65,100)
	test_samples.append(random_older)
	test_labels.append(1)
test_labels=np.array(test_labels)
test_samples=np.array(test_samples)

# scaler=MinMaxScaler(feature_range=(0,1))
scaled_test_samples=scaler.fit_transform((test_samples).reshape(-1,1))
#Predictions

# predictions=model.predict(scaled_test_samples,batch_size=10,verbose=0)
# for i in predictions:
# 	print (i)

rounded_predictions=model.predict_classes(scaled_test_samples,batch_size=10,verbose=0)
for i in rounded_predictions:
	print (i)

#Confusion matrix

cm=confusion_matrix(test_labels,rounded_predictions)

def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
	plt.imshow(cm,interpolation='nearest',cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks=np.arange(len(classes))
	plt.xticks(tick_marks,classes,rotation=45)
	plt.yticks(tick_marks,classes)

	if normalize:
		cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion Matrix,without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
		plt.text(j,i,cm[i,j],horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
	plt.tight_layout()
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')
cm_plot_labels=['no_side_effects','had_side_effects']
plot_confusion_matrix(cm,cm_plot_labels,title='Confusion Matrix')
plt.show()

model.save('medical_trial_model.h5')
