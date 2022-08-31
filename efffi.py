import os
import sys
import numpy as np
import cv2
from time import sleep
from tqdm import tqdm,trange
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils
import matplotlib.pyplot as plt
import tensorflow as tf
###
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

###
##set input image size
input_shape = (300,300,3)
IMAGE_SIZE = 300
#resize
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #get size
    h, w, _ = image.shape
    
    #adj(w,h)
    longest_edge = max(h, w)    
    
    #size = n*n 
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 

    BLACK = [0, 0, 0]   
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    return cv2.resize(constant, (height, width))
    
##################### 改這裡 ####################
#load_data
# OK樣本路徑
A = "D:\\harden\\dataset\\L128 Mura\\OK\\"
# NG樣本路徑
B = "D:\\harden\\dataset\\L128 Mura\\NG\\"
#fall1 = "C:\\Users\\User.DESKTOP-IIINHE5\\Desktop\\fall1_img"

images = []
labels = []
dir_counts = 0
vou=0
##################### 改這裡 ####################
for i in tqdm(os.listdir(A)):
    if i.split('.')[-1]=='png':
        sleep(0.005)
        img1 = cv2.imread(A+i)
        img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
        #img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
        images.append(img1)
        labels.append(dir_counts)
        vou +=1
        if vou >=200:
            break
    #a = np.array(images,dtype=np.float32)
    #print(a.shape)
vou=0
for i in tqdm(os.listdir(B)):
    if i.split('.')[-1]=='png':
        sleep(0.005)
        img2 = cv2.imread(B+i)
        img2 = resize_image(img2, IMAGE_SIZE, IMAGE_SIZE)
        #img2 = cv2.resize(img2,(IMAGE_SIZE,IMAGE_SIZE))
        images.append(img2)
        labels.append(dir_counts+1)
        vou +=1
        if vou >=600:
            break
    #a = np.array(images,dtype=np.float32)
    #print(a.shape)

label = np.array(labels)
#########Train/Test############
X_train_img,X_test_img,y_train_label,y_test_label =  train_test_split(images, label,test_size=0.4,random_state=2 )#
X_train = np.array(X_train_img, dtype=np.float32)
X_test = np.array(X_test_img, dtype=np.float32)
#print("X_train.shape",X_train.shape)
x_train_std = X_train/255.0
x_test_std  =  X_test/255.0
y_trainOneHot = np_utils.to_categorical(y_train_label)
y_testOneHot = np_utils.to_categorical(y_test_label)
print("x_test_std.shape",x_test_std.shape)
print("y_test_label",y_test_label.shape)
ok_count=0
ng_count=0
for k in range(len(y_train_label)):
    if y_train_label[k]==1:
        ng_count+=1
    else:
        ok_count+=1
print("ok counts= ", ok_count,"ng counts= ", ng_count)
################################


# Set the augmentation parameters and fit the training data
############### change here #################
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.0,
    fill_mode="constant",
    cval=0.2
    
    
)
############### change here #################
datagen.fit(x_train_std)


#Model
'''
from tensorflow.keras.applications import MobileNet
base_model = MobileNet(
    input_shape=input_shape,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000
)
'''
from tensorflow.keras.applications import EfficientNetB0
base_model=EfficientNetB0(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=input_shape,#None,
    pooling=None,
    classes=1000,
    classifier_activation='sigmoid'#,softmax
    #include_preprocessing=True
)



x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(2, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions) #inputs=base_model.input

print("summary: ", model.summary())

# Compile the model
model.compile(
      optimizer=keras.optimizers.Adam(3e-4),
      loss="binary_crossentropy",
      metrics=["accuracy"],
)
#summary
#model.summary()

# Set the epochs and batch size, then train the model
############### change here #################
epochs = 50
batch_size = 16
print("x_train_std",len(x_train_std))
print("x_test_std",len(x_test_std))
############### change here #################
directory='./'
history = model.fit(
    datagen.flow(x_train_std, y_trainOneHot,shuffle=True, batch_size=batch_size,save_format='png'),
    steps_per_epoch=len(X_train)/batch_size,
    epochs=epochs,
    validation_data=(x_test_std, y_testOneHot)
)#,save_to_dir='./dan'

history_dict = history.history
print(history_dict.keys())
import matplotlib.pyplot as plt
# Plot loss and accuracy
def plt_loss(history):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left') 
    #plt.show()
    # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left') 
    plt.show()
plt_loss(history)

model.save('weights/effibMU.h5')
print("model save")

model.load_weights('weights/effibMU.h5')
#Predict
predict_y = model.predict(x_test_std)



#confusion_matrix
predict_y[predict_y >= 0.5] = 1
predict_y[predict_y < 0.5] = 0
print(confusion_matrix(y_testOneHot.argmax(axis=1), predict_y.argmax(axis=1), labels=[1, 0]))

y=y_testOneHot.argmax(axis=1)
p_y=predict_y.argmax(axis=1)
# Calculate the sensitivity and specificity
TP = confusion_matrix(y, p_y, labels=[1, 0])[0, 0]
FP = confusion_matrix(y, p_y, labels=[1, 0])[1, 0]
FN = confusion_matrix(y, p_y, labels=[1, 0])[0, 1]
TN = confusion_matrix(y, p_y, labels=[1, 0])[1, 1]
print("True positive: {}".format(TP))
print("False positive: {}".format(FP))
print("False negative: {}".format(FN))
print("True negative: {}".format(TN))
############################
sensitivity = TP/(FN+TP)
specificity = TN/(TN+FP)
recall= TP/(TP+FP)
precision=TP/(TP+FP)
################################
print("Sensitivity: {}".format(sensitivity))
print("Specificity: {}".format(specificity))
print("recall: {}".format(recall))
print("precisionl: {}".format(precision))


# Plot the ROC curve of the test results
def plt_auc(y_test_label,predict_y):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')

    fpr, tpr, _ = roc_curve(y_test_label, predict_y)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = {}'.format(roc_auc))

    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
plt_auc(y,predict_y.argmax(axis=1))

def plot_image(image,labels,prediction,idx,num=10):  
    fig = plt.gcf() 
    fig.set_size_inches(12, 14) 
    if num>25: 
        num=25 
    for i in range(0, num): 
        ax = plt.subplot(5,5, 1+i) 
        ax.imshow(image[idx], cmap='binary') 
        title = "label=" +str(labels[idx]) 
        if len(prediction)>0: 
            title+=",perdict="+str(prediction[idx]) 
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([]) 
        idx+=1 
    plt.show() 
plot_image(x_test_std,y_test_label,predict_y.argmax(axis=1),idx=10)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

import sklearn.metrics as metrics
import itertools
cnf_matrix = metrics.confusion_matrix(y_test_label,predict_y.argmax(axis=1))
target_names = ['NG', 'OK']
# plot_confusion_matrix(predict_y)
plot_confusion_matrix(cnf_matrix, classes=target_names)