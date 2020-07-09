#!/usr/bin/env python
# coding: utf-8

# In[38]:


#Image Classification - classifying the sewer images as partially blocked, fully blocked, normal and cracked



#Importing Modules
import numpy as np
import pandas as pd
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image, ImageChops
import glob


# In[56]:


#Reading the data from dataset
df = pd.DataFrame(columns=['image_name', 'image_path', 'label', 'width', 'height'])

normal_img_path = 'C:\\Users\\DELL1\\Desktop\\Robotic Vision Pre Dataset\\no blockage\\'
cracks_img_path = 'C:\\Users\\DELL1\\Desktop\\Robotic Vision Pre Dataset\\cracks images\\'
f_block_img_path = 'C:\\Users\\DELL1\\Desktop\\Robotic Vision Pre Dataset\\fully blocked\\'
p_block_img_path = 'C:\\Users\\DELL1\\Desktop\\Robotic Vision Pre Dataset\\partially blocked\\'

normal_imgs = os.listdir(normal_img_path)
cracks_imgs = os.listdir(cracks_img_path)
f_block_imgs = os.listdir(f_block_img_path)
p_block_imgs = os.listdir(p_block_img_path)

count_value = 0

for img_name in normal_imgs:
    df.loc[count_value, ['image_name']] = img_name
    df.loc[count_value, ['image_path']] = normal_img_path + img_name
    df.loc[count_value, ['label']] = 'normal'
    
    img = plt.imread(normal_img_path + img_name)
    df.loc[count_value, ['width']] = img.shape[0]
    df.loc[count_value, ['height']] = img.shape[1]
    
    count_value += 1
    
    
for img_name in cracks_imgs:
    df.loc[count_value, ['image_name']] = img_name
    df.loc[count_value, ['image_path']] = cracks_img_path + img_name
    df.loc[count_value, ['label']] = 'cracks'
    
    img = plt.imread(cracks_img_path + img_name)
    df.loc[count_value, ['width']] = img.shape[0]
    df.loc[count_value, ['height']] = img.shape[1]
    
    count_value += 1
    

for img_name in f_block_imgs:
    df.loc[count_value, ['image_name']] = img_name
    df.loc[count_value, ['image_path']] = f_block_img_path + img_name
    df.loc[count_value, ['label']] = 'full_block'
    
    img = plt.imread(f_block_img_path + img_name)
    df.loc[count_value, ['width']] = img.shape[0]
    df.loc[count_value, ['height']] = img.shape[1]
    
    count_value += 1


for img_name in p_block_imgs:
    df.loc[count_value, ['image_name']] = img_name
    df.loc[count_value, ['image_path']] = p_block_img_path + img_name
    df.loc[count_value, ['label']] = 'partial_block'
    
    img = plt.imread(p_block_img_path + img_name)
    df.loc[count_value, ['width']] = img.shape[0]
    df.loc[count_value, ['height']] = img.shape[1]
    
    count_value += 1


# In[58]:


df.head()


# In[59]:


sns.countplot(df['label'])


# In[43]:


# Code for black border trimming 
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        return bg


# In[44]:


#for partial block

image_list1 = []
resized_images1 = []
denoised_images1=[]
resized1=[]

for filename in glob.glob('C:\\Users\\DELL1\\Desktop\\Robotic Vision Pre Dataset\\partially blocked\\*.jpg'):
    img = Image.open(filename)
    image = trim(img)
    image_list1.append(image)

#Resizing
for image in image_list1:
    image = image.resize((480, 320))
    resized_images1.append(image)

for (i, new) in enumerate(resized_images1):
    new.save('C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\Partial Block\\'+str(i+1)+'.jpg')

    
for i in range(0,112):
    img=cv2.imread('C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\Partial Block\\'+str(i+1)+'.jpg')
    
    #Gaussian Blur
    blur = cv2.GaussianBlur(img,(5,5),0)
    
    resized1.append(img)
    denoised_images1.append(blur)  
    cv2.imwrite('C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\Partial Block\\'+str(i+1)+'.jpg',blur)

cv2.imshow('Unblurred',resized1[106])
cv2.imshow('Blurred',denoised_images1[106])
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[46]:


#for full block

image_list2 = []
resized_images2 = []
denoised_images2=[]
resized2=[]

for filename in glob.glob('C:\\Users\\DELL1\\Desktop\\Robotic Vision Pre Dataset\\fully blocked\\*.jpg'):
    img = Image.open(filename)
    image = trim(img)
    image_list2.append(image)

#Resizing
for image in image_list2:
    image = image.resize((480, 320))
    resized_images2.append(image)

for (i, new) in enumerate(resized_images2):
    new.save('C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\Full Block\\'+str(i+1)+'.jpg')

    
for i in range(len(resized_images2)):
    img=cv2.imread('C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\Full Block\\'+str(i+1)+'.jpg')
    
    #Gaussian Blur
    blur = cv2.GaussianBlur(img,(5,5),0)
    
    resized2.append(img)
    denoised_images2.append(blur)  
    cv2.imwrite('C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\Full Block\\'+str(i+1)+'.jpg',blur)

cv2.imshow('Unblurred',resized2[10])
cv2.imshow('Blurred',denoised_images2[10])
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[47]:


#for normal

image_list3 = []
resized_images3 = []
denoised_images3=[]
resized3=[]

for filename in glob.glob('C:\\Users\\DELL1\\Desktop\\Robotic Vision Pre Dataset\\no blockage\\*.jpg'):
    img = Image.open(filename)
    image = trim(img)
    image_list3.append(image)

#Resizing
for image in image_list3:
    image = image.resize((480, 320))
    resized_images3.append(image)

for (i, new) in enumerate(resized_images3):
    new.save('C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\Normal\\'+str(i+1)+'.jpg')

    
for i in range(len(resized_images3)):
    img=cv2.imread('C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\Normal\\'+str(i+1)+'.jpg')
    
    #Gaussian Blur
    blur = cv2.GaussianBlur(img,(5,5),0)
    
    resized3.append(img)
    denoised_images3.append(blur)  
    cv2.imwrite('C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\Normal\\'+str(i+1)+'.jpg',blur)

cv2.imshow('Unblurred',resized3[10])
cv2.imshow('Blurred',denoised_images3[10])
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[48]:


#for crack

image_list4 = []
resized_images4 = []
denoised_images4 =[]
resized4 =[]

for filename in glob.glob('C:\\Users\\DELL1\\Desktop\\Robotic Vision Pre Dataset\\cracks images\\*.jpg'):
    img = Image.open(filename)
    image = trim(img)
    image_list4.append(image)

#Resizing
for image in image_list4:
    image = image.resize((480, 320))
    resized_images4.append(image)

for (i, new) in enumerate(resized_images4):
    new.save('C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\Cracks\\'+str(i+1)+'.jpg')

    
for i in range(len(resized_images4)):
    img=cv2.imread('C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\Cracks\\'+str(i+1)+'.jpg')
    
    #Gaussian Blur
    blur = cv2.GaussianBlur(img,(5,5),0)
    
    resized4.append(img)
    denoised_images4.append(blur)  
    cv2.imwrite('C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\Cracks\\'+str(i+1)+'.jpg',blur)

cv2.imshow('Unblurred',resized4[9])
cv2.imshow('Blurred',denoised_images4[9])
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[52]:


from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# create a new generator
imagegen = ImageDataGenerator()
# load train data
train = imagegen.flow_from_directory("C:\\Users\\DELL1\\Desktop\\Robotic Vision Post Dataset\\", class_mode="categorical", shuffle=False, batch_size=10, target_size=(480, 320))
# load val data
val = imagegen.flow_from_directory("C:\\Users\\DELL1\\Desktop\\Val\\", class_mode="categorical", shuffle=False, batch_size=5, target_size=(480, 320))


# In[54]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

# build a sequential model
model = Sequential()
model.add(InputLayer(input_shape=(480, 320, 3)))

# 1st conv block
model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
# 2nd conv block
model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())
# 3rd conv block
model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())
# ANN block
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.25))
# output layer
model.add(Dense(units=4, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
# fit on data for 30 epochs
model.fit_generator(train, epochs=30, validation_data=val)


# In[ ]:




