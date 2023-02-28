

#Part 1- Building the CNN as preprocessing is done beforehand

#importing keras libraries and packages

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier=Sequential()

#Step 1 COnvolution

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
 #32 feature detectors of 3x3 layers and ip shape has 3 since it's a colored image
 
 #Step 2 Pooling
 
 classifier.add(MaxPooling2D(pool_size=(2,2)))
 
 #Adding a second convolutional layer. Here, the i/p shape need not be given as keras knows.
 
classifier.add(Convolution2D(32,3,3,activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

 #Step 3 Flattening
 
 classifier.add(Flatten())
 
 #Step 4 Full COnnection
 
 classifier.add(Dense(output_dim= 128, activation= 'relu'))
 classifier.add(Dense(output_dim= 1, activation= 'sigmoid'))
 
 #Compiling the CNN
 
 classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics=['accuracy'])
 
 #Part 2- Fitting the CNN to the images
 
 from keras.preprocessing.image import ImageDataGenerator
 
 
 train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

#This will take a long time to execute . Upon ompletion, we find an accuracy
#of 84% on th training set and 75% on testing set

#To improve performance, we either add another convolutional layer or another
 #fully connected layer. Usually, we add another convo. layer. This is now node 
 #above.
 
 #By doing this, we observe that we obtain an acuracy of 85% on the training set
 #and 81% on the testing set
 
 #Part 3 Making new predictions
 
import numpy as np #For preprocessing the image
from keras.preprocessing import image
 
test_image=image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64)) 
 #Target size must be same as that of training set 
 
 #We must add a new dimension to test_image since the ip image in the ip layer of our CNN
 #has 3 dimensions (64,64,3). 3 since colored image
 
 #Adding the third dimension,
 
 test_image=image.img_to_array(test_image)
 
 #We must add one more dimension since if we use the predict method directly, 
 #it'll throw up an error
 
 #The new dimension corresponds to the batch. In general , the functions in neural networlks
 #like predict function, cannot accept a single image by itselt. It only accepts batch.
 
 #We add the dim using expand_dims
 
 test_image=np.expand_dims(test_image,axis=0) #axis =0 means the index of the new dimesnion
 #that we'e adding will have the first index ie, index 0
 
result=classifier.predict(test_image)
 
#The result will be 1 or 0

training_set.class_indeces #GIves what 1 and 0 corresponds to


 
