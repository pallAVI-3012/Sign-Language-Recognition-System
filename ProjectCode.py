#!/usr/bin/env python
# coding: utf-8

# # Loading the data and data preprocessing
# 

# In[1]:


#important libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the training dataset
data_train = pd.read_csv(r"C:\Users\KIIT\OneDrive\Desktop\sign_mnist_train.csv")

#rows and cols of training dataset
print(data_train.shape)

#important features of training dataset
data_train.describe()


# In[3]:


#display columns
data_train.columns


# In[4]:


#display 5 rows of training dataset
data_train.head()


# In[5]:


# Load the testing dataset
data_test = pd.read_csv(r"C:\Users\KIIT\OneDrive\Desktop\sign_mnist_test.csv")

#rows and cols of testing dataset
print(data_test.shape)

#important features of testing dataset
data_test.describe()


# In[33]:


#display 5 rows of training dataset
data_test.head()


# In[6]:


unique_labels=data_train['label'].unique()
label_indices = {}
for label in unique_labels:
    first_index = data_train.index[data_train['label'] == label][0]
    label_indices[label] = first_index

# Print the label and its corresponding first occurrence index
for label, first_index in label_indices.items():
    print(f"Label {label}: First Occurrence Index {first_index}")


# # Data Preprocessing

# In[7]:


# extract labels from training dataframe and converts into 1D array
#image data excluding labels,reshapes in 28x28 pixel images and converts in 4D array with shape (-1,28,28,1)

labels_train = data_train['label'].values
images_train = data_train.drop('label', axis=1).values.reshape(-1, 28, 28,1)

print("Shape of labels_train:", labels_train.shape)
print("Shape of imagess_train:", images_train.shape)


# In[8]:


# extract labels from testing dataframe and converts into 1D array
#image data excluding labels,reshapes in 28x28 pixel images and converts in 4D array with shape (-1,28,28,1)

labels_test = data_test['label'].values
images_test = data_test.drop('label', axis=1).values.reshape(-1, 28, 28,1)

print("Shape of labels_test:", labels_test.shape)
print("Shape of images_test:", images_test.shape)


# In[9]:


# Normalize pixel values of training data to [0, 1]
preprocessed_images_train = []

for image in images_train:
    normalized_image = image / 255.0  
    preprocessed_images_train.append(normalized_image)

preprocessed_images_train = np.array(preprocessed_images_train)


# In[38]:


# Normalize pixel values of testing data to [0, 1]
preprocessed_images_test = []

for image in images_test:
    normalized_image = image / 255.0  
    preprocessed_images_test.append(normalized_image)

preprocessed_images_test = np.array(preprocessed_images_test)


# # Data Visualization

# In[10]:


fig, ax = plt.subplots(8, 3, figsize=(10, 10))
fig.suptitle('Preview of dataset')
ax[0,0].imshow(images_train[46].reshape(28,28),cmap='gray')
ax[0,0].set_title('label: 0  letter: A',fontsize=15)
ax[0,1].imshow(images_train[29].reshape(28,28),cmap='gray')
ax[0,1].set_title('label: 1  letter: B',fontsize=15)
ax[0,2].imshow(images_train[2].reshape(28,28),cmap='gray')
ax[0,2].set_title('label: 2  letter: C',fontsize=15)
ax[1,0].imshow(images_train[0].reshape(28,28),cmap='gray')
ax[1,0].set_title('label: 3  letter: D',fontsize=15)
ax[1,1].imshow(images_train[44].reshape(28,28),cmap='gray')
ax[1,1].set_title('label: 4  letter: E',fontsize=15)
ax[1,2].imshow(images_train[48].reshape(28,28),cmap='gray')
ax[1,2].set_title('label: 5  letter: F',fontsize=15)
ax[2,0].imshow(images_train[1].reshape(28,28),cmap='gray')
ax[2,0].set_title('label: 6  letter: G',fontsize=15)
ax[2,1].imshow(images_train[49].reshape(28,28),cmap='gray')
ax[2,1].set_title('label: 7  letter: H',fontsize=15)
ax[2,2].imshow(images_train[6].reshape(28,28),cmap='gray')
ax[2,2].set_title('label: 8  letter: I',fontsize=15)
ax[3,0].imshow(images_train[11].reshape(28,28),cmap='gray')
ax[3,0].set_title('label: 10  letter: K',fontsize=15)
ax[3,1].imshow(images_train[40].reshape(28,28),cmap='gray')
ax[3,1].set_title('label: 11  letter: L',fontsize=15)
ax[3,2].imshow(images_train[31].reshape(28,28),cmap='gray')
ax[3,2].set_title('label: 12  letter: M',fontsize=15)
ax[4,0].imshow(images_train[4].reshape(28,28),cmap='gray')
ax[4,0].set_title('label: 13  letter: N',fontsize=15)
ax[4,1].imshow(images_train[61].reshape(28,28),cmap='gray')
ax[4,1].set_title('label: 14  letter: O',fontsize=15)
ax[4,2].imshow(images_train[41].reshape(28,28),cmap='gray')
ax[4,2].set_title('label: 15  letter: P',fontsize=15)
ax[5,0].imshow(images_train[5].reshape(28,28),cmap='gray')
ax[5,0].set_title('label: 16  letter: Q',fontsize=15)
ax[5,1].imshow(images_train[16].reshape(28,28),cmap='gray')
ax[5,1].set_title('label: 17  letter: R',fontsize=15)
ax[5,2].imshow(images_train[10].reshape(28,28),cmap='gray')
ax[5,2].set_title('label: 18  letter: S',fontsize=15)
ax[6,0].imshow(images_train[19].reshape(28,28),cmap='gray')
ax[6,0].set_title('label: 19  letter: T',fontsize=15)
ax[6,1].imshow(images_train[14].reshape(28,28),cmap='gray')
ax[6,1].set_title('label: 20  letter: U',fontsize=15)
ax[6,2].imshow(images_train[21].reshape(28,28),cmap='gray')
ax[6,2].set_title('label: 21  letter: V',fontsize=15)
ax[7,0].imshow(images_train[7].reshape(28,28),cmap='gray')
ax[7,0].set_title('label: 22  letter: W',fontsize=15)
ax[7,1].imshow(images_train[23].reshape(28,28),cmap='gray')
ax[7,1].set_title('label: 23  letter: X',fontsize=15)
ax[7,2].imshow(images_train[26].reshape(28,28),cmap='gray')
ax[7,2].set_title('label: 24  letter: Y',fontsize=15)


# In[12]:


# Analyze the distribution of labels in training dataset
distribution_train = data_train['label'].value_counts().sort_index()
print(distribution_train)


# In[11]:


#plotting the training quantities in each label
plt.figure(figsize=(18,8))
sns.countplot(x=labels_train)


# In[12]:


# Analyze the distribution of labels in training dataset
distribution_test = data_test['label'].value_counts().sort_index()
print(distribution_test)


# In[42]:


#plotting the testing quantities in each label
plt.figure(figsize=(18,8))
sns.countplot(x=labels_test)


# # Building The CNN Model

# In[16]:


#importing important CNN Model libraries

import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from tensorflow.keras.models import save_model, load_model


# In[17]:


model=Sequential()
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
          
model.add(Flatten())


# In[28]:


model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24,activation='softmax'))
model.summary()


# In[18]:


#initializing a sequential model
#creating a linear stack of layers
model=Sequential()

#Convolutional layer 1 -- 128 KERNEL SIZE - 5 * 5 -- STRIDE LENGTH - 1 -- ACTIVATION - ReLu
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))

#MaxPool layer 1 -- MAX POOL WINDOW - 3 * 3 -- STRIDE - 2
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))

#Convolutional layer 2 -- 64 KERNEL SIZE - 3 * 3 -- STRIDE LENGTH - 1 -- ACTIVATION - ReLu
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))

#MaxPool layer 2 -- MAX POOL WINDOW - 2 * 2 -- STRIDE - 2
model.add(MaxPool2D((2,2),2,padding='same'))

#Convolutional layer 3 -- 32 KERNEL SIZE - 2 * 2 -- STRIDE LENGTH - 1 -- ACTIVATION - ReLu
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))

#MaxPool layer 3 -- MAX POOL WINDOW - 2 * 2 -- STRIDE - 2
model.add(MaxPool2D((2,2),2,padding='same'))
          
#converts the output of the previous layer into a one-dimensional vector
model.add(Flatten())

#It adds a fully connected (dense) layer to the model. 
#It has 512 neurons, and the ReLU activation function is applied to its output.
model.add(Dense(units=512,activation='relu'))

#It adds a dropout layer to the model. 
#Dropout is a regularization technique used to prevent overfitting by randomly setting a fraction of input units to zero during training.
model.add(Dropout(rate=0.25))

#It adds another fully connected layer to the model, with 24 neurons corresponding to the number of classes. 
#The softmax activation function is applied to its output, which outputs a probability distribution over the classes.
model.add(Dense(units=1,activation='softmax'))

#It includes the architecture of the model and the number of parameters in each layer.
model.summary()


# # Training the model

# In[19]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(images_train,labels_train,batch_size=200,
         epochs = 35,
          validation_data=(images_test,labels_test),
          shuffle=1
         )


# In[20]:


(ls,acc)=model.evaluate(x=images_test,y=labels_test)


# In[21]:


print('MODEL ACCURACY = {}%'.format(acc*100))


# In[22]:


model.save("model.keras")


# In[23]:


(ls1,acc1)=model.evaluate(images_test,labels_test,batch_size=200)


# In[24]:


print('TEST ACCURACY = {}%'.format(acc1*100))


# # Making Predictions

# In[26]:


import numpy as np

# Assuming images_test[idx2, :] has shape (28, 28)
input_data = np.expand_dims(images_test[idx2, :], axis=0)  # Add a batch dimension
input_data = np.expand_dims(input_data, axis=-1)  # Add a channel dimension

# Make predictions
y_pred = model.predict(input_data)

# Optionally, you can print the shape of the input_data and y_pred to verify
print("Shape of input_data:", input_data.shape)
print("Shape of y_pred:", y_pred.shape)


# In[28]:


import random 
import matplotlib.pyplot as plt

idx2 = random.randint(0, len(labels_test))
plt.imshow(images_test[idx2])
plt.title(f'Label: {labels_test[idx2]}')  # Display the label as the title
plt.show()

# Assuming your model expects input shape (28, 28, 1)
y_pred = model.predict(images_test[idx2].reshape(1, 28, 28, 1))


# In[ ]:


import random
import matplotlib.pyplot as plt
from skimage.transform import resize

idx2 = random.randint(0, len(labels_test))
resized_image = resize(images_test[idx2], (500, 500))  # Resize the image to (10, 10)
plt.imshow(resized_image)
plt.title(f'Label: {labels_test[idx2]}')  # Display the label as the title
plt.show()

# Assuming your model expects input shape (28, 28, 1)
y_pred = model.predict(resized_image.reshape(1, 500, 500, 1))


# In[ ]:




