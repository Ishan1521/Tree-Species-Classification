#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Necessary Libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from random import choice, shuffle
from matplotlib import pyplot as plt
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')


# In[2]:


now = datetime.now().strftime("%m-%d-%Y-%H-%M-%S") # current date and time
print(now)


# In[3]:


root_dir = '/kaggle/working'
output_dir = f'{root_dir}/output/{now}'
#data_dir = Your Dataset


# In[4]:


os.listdir(data_dir)


# In[5]:


os.makedirs(output_dir)


# In[6]:


data = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size=32, image_size=(224, 224))


# In[7]:


label_classes = data.class_names
num_classes = len(data.class_names)
print(label_classes)


# In[8]:


data_iterator = data.as_numpy_iterator()
images, labels = next(data_iterator)


# In[9]:


figure, axs = plt.subplots(len(images)//4,4, figsize=(10, 10))
axs = axs.ravel()

for idx, batch in enumerate(zip(images,labels)):
    image, label = batch
    axs[idx].imshow(image.astype('uint8'))
    axs[idx].set_title(f"{label_classes[label]} : {label}")
    axs[idx].axis('off')
plt.tight_layout()
plt.show()

#plt.savefig(f'{output_dir}/sample_data_label.png')


# In[10]:


#Splitting Data
s_data = data.map(lambda x,y: (x/255, y))
train_size = int(len(s_data)*.7)
val_size = int(len(s_data)*.2)
test_size = int(len(s_data)*.1) 
print(train_size + val_size + test_size, len(s_data))


# In[11]:


train = s_data.take(train_size)
val = s_data.skip(train_size).take(val_size)
test = s_data.skip(train_size+val_size).take(test_size)


# In[12]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam


# In[13]:


#CNN Model
model = Sequential()
model.add(Conv2D(filters=64,padding='same',strides=2,kernel_size=3,activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(filters=32,padding='same',strides=2,kernel_size=3,activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(filters=32,padding='same',strides=2,kernel_size=3,activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# In[14]:


model.summary()


# In[ ]:





# In[ ]:





# In[15]:


# Define the optimizer with a custom learning rate
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()


# In[16]:


from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint


# In[17]:


logdir=f'{output_dir}/logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1)
checkpoint = ModelCheckpoint(os.path.join('models',f'{now}-imageclassifier.h5'), monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
es = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')


# In[ ]:





# In[18]:


# Define the optimizer with a custom learning rate
#learning_rate = 0.003
#optimizer = Adam(learning_rate=learning_rate)
#model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
#model.summary()


# In[ ]:


hist = model.fit(train, epochs=32, validation_data=val, callbacks=[tensorboard_callback,es, checkpoint])


# In[ ]:


plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
#plt.savefig(f'{output_dir}/loss.png')
 
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()
# plt.savefig(f'{output_dir}/accuracy.png')


# In[ ]:


from tensorflow.keras.metrics import  SparseCategoricalAccuracy
accuracy = SparseCategoricalAccuracy()


# In[ ]:


y_test, predictions = np.array([]), np.array([])
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    accuracy.update_state(y, yhat)
    y_test = np.concatenate((y_test, y))
    predictions = np.concatenate((predictions, np.argmax(yhat, axis=1)))
if y_test.shape != predictions.shape:
    if y_test.shape > predictions.shape:
        y_test = y_test[:predictions.shape[0]]
    if y_test.shape < predictions.shape:
        predictions = predictions[:y_test.shape[0]]    
print(f"Accuracy : {(accuracy.result().numpy() * 100):.2f} %")

y_test.shape, predictions.shape


# In[ ]:


from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


confusion_matrix = confusion_matrix(y_test, predictions)

#cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

#cm_display.plot()
plt.show()
import seaborn as sns

#Plot the confusion matrix.
sns.heatmap(confusion_matrix,
            annot=True,
            cmap='rocket',
            fmt='g',
            xticklabels=list(label_classes),
            yticklabels=list(label_classes))
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# In[ ]:


os.path.join('models',f'{now}-3 cnn.h5')


# In[ ]:


model.save('3 cnn.h5')


# In[ ]:


import os
os.chdir(r'/kaggle/working')

get_ipython().system('zip imageclassifier.h5.zip imageclassifier.h5')

from IPython.display import FileLink

FileLink(r'3 cnn.h5.zip')


# In[ ]:




