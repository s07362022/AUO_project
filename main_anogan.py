# from __future__ import print_function

import matplotlib
matplotlib.use('agg')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from keras.datasets import mnist
import argparse
import anogan2 as anogan
import dist 
import tensorflow as tf
import itertools
import sklearn.metrics as metrics

####GPU

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=5)
parser.add_argument('--label_idx', type=int, default=0)
parser.add_argument('--mode', type=str, default='test', help='train, test')
args = parser.parse_args()


IMAGE_SIZE = 300#300
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #get size
    h, w = image.shape
    
    longest_edge = max(h, w)    
   
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

### data 
D = "D:\\harden\\dataset\\H1_water_de\\3ok\\"#"D:\\harden\\dataset\\T85N3 RB\\OK\\"#"D:\\harden\\dataset\\H1_water_de\\3ok\\"##"D:\\harden\\dataset\\L128 Mura\\OK\\"#"D:\\harden\\dataset\\H1_water_de\\3ok\\"   #####path  ok
E = "D:\\harden\\dataset\\H1_water_de\\3ng\\"#"D:\\harden\\dataset\\T85N3 RB\\NG\\"#"D:\\harden\\dataset\\H1_water_de\\3ng\\" ##"D:\\harden\\dataset\\L128 Mura\\NG\\"#"D:\\harden\\dataset\\H1_water_de\\3ng\\"   #####path  ng
images = []
labels = []
test_image = []
test_label = []
dir_counts = 0
IMAGE_SIZE = 300#300
###   vou 指需讀取影像的數量，可修改 ###

def d (D=D,images=images,labels=labels):
    vou=0
    for i in os.listdir(D):
        img1 = cv2.imread(D+i,0)
        img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
        images.append(img1)
        labels.append(dir_counts)
        
        vou +=1
        if vou >=600: 
            break
    print("OK train already read")
    return(images,labels)
def d2 (D=D,images=images,labels=labels):
    vou=0
    for i in os.listdir(D):
        img1 = cv2.imread(D+i,0)
        img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
        images.append(img1)
        labels.append(dir_counts+1)
        vou +=1
        if vou >=30:
            break
    print("NG test already read")
    return(images,labels)
 
def d3 (D=D,images=images,labels=labels):
    vou=0
        
    for i in os.listdir(D):
        try:
            img1 = cv2.imread(D+i,0)
            img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img1)
            labels.append(dir_counts)
        except:
            print("error")
        vou +=1
        if vou >=30:
            break
    print("OK test already read")
    return(images,labels)
d(D,images,labels)
d2(E,images=test_image,labels=test_label)
d3(D,images=test_image,labels=test_label)
label = np.array(labels)
images = np.array(images)
test_imgae = np.array(test_image)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =  train_test_split(images, label,test_size=0.1,random_state=42 )#

X_train = (X_train.astype(np.float32) /255.0) #- 127.5) / 127.5
X_test = (test_imgae.astype(np.float32) /255.0) 
test_label= np.array(test_label)
X_train = X_train[:,:,:,None]
X_test = X_test[:,:,:,None]
X_train_testing = X_train[:len(X_test)]  #######################
X_test_original = X_test.copy()

print ('train shape:', X_train.shape)
print ('test shape:', X_test.shape)
### 1. train generator & discriminator
if args.mode == 'train':
    Model_d, Model_g = anogan.train(20, X_train) #32

### 2. test generator
generated_img = anogan.generate(25)
img = anogan.combine_images(generated_img)
img = (img*127.5)+127.5
img = img.astype(np.uint8)
img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


### 3. other class anomaly detection
diff_list=[]
sim_list=[]
def anomaly_detection(test_img, g=None, d=None):
    model = anogan.anomaly_detector(g=g, d=d)
    # ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1), iterations=500, d=d)
    ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 300, 300, 1), iterations=500, d=d)

    # anomaly area, 255 normalization
    # np_residual = test_img.reshape(28,28,1) - similar_img.reshape(28,28,1)
    np_residual = test_img.reshape(300,300,1) - similar_img.reshape(300,300,1) #.reshape(300,300,1)
    np_residual = (np_residual + 2)/2

    np_residual = (255*np_residual).astype(np.uint8)
    # original_x = (test_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)
    #　similar_x = (similar_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)
    original_x = (test_img.reshape(300,300,1)*127.5+127.5).astype(np.uint8)#.reshape(300,300,1)
    similar_x = (similar_img.reshape(300,300,1)*127.5+127.5).astype(np.uint8)#.reshape(300,300,1)

    original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
    residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_COOL) #COLORMAP_RAINBOW COLORMAP_JET
    show = cv2.addWeighted(original_x_color, 0.3, residual_color, 0.7, 0.)
    diff_list.append(show)
    sim_list.append(similar_x)
    return ano_score, original_x, similar_x, show




### compute anomaly score - sample from strange image
img_idx = args.img_idx
label_idx = args.label_idx
test_img = X_test_original[img_idx]
# test_img = np.random.uniform(-1,1, (28,28,1))
def polt_re(qurey,pred,diff,count):
    fig=plt.figure()
    plt.subplot(1,3,1)
    plt.title('query image')
    plt.imshow(qurey.reshape(300,300), cmap=plt.cm.gray)# .reshape(300,300)
    plt.subplot(1,3,2)
    plt.title('generated similar image')
    plt.imshow(pred.reshape(300,300), cmap=plt.cm.gray) #.reshape(300,300)
    plt.subplot(1,3,3)
    plt.title('anomaly detection')
    plt.imshow(cv2.cvtColor(diff,cv2.COLOR_BGR2RGB))
    # 儲存位置
    fig.savefig('./fake_img/0817/output_{}.png'.format(count),dpi=fig.dpi)
    #plt.show()
print("len(X_test_original): ",len(X_test_original))
# score, qurey, pred, diff = anomaly_detection(test_img)


###########################################################
scorelist2= []
y_train = y_train[:len(X_test_original)]
for i in range(len(X_train_testing)):
    # start = cv2.getTickCount()
    score, qurey, pred, diff = anomaly_detection(X_train_testing[i])    
    scorelist2.append(score)
    # time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print ('%d label, %d : done'%(y_train[i], img_idx), 'score: %.2f '%score)#, '%.2fms'%time)
    count = str(i)
    # 儲存位置
    #cv2.imwrite('./fake_img/0812_ori/tarin_gqurey{}.png'.format(count) , qurey)
    #cv2.imwrite('./fake_img/0812_ori/tarin_pred{}.png'.format(count), pred)
    #cv2.imwrite('./fake_img/0812_ori/tarin_diff{}.png'.format(count), diff)
    #polt_re(qurey,pred,diff,count)
print("訓練集分數: ",scorelist2)


import pandas as pd
name_list = scorelist2
height_list = y_train
df_train = pd.DataFrame((zip(name_list, height_list)), columns = ['score', 'labe'])
print(df_train)
df_train.to_csv("./anogan_0817_PRtrain.csv")#scorelist

########### use train score to set gate ############
#try:
    #df_train=df_train.drop(['Unnamed: 0'],axis=1)
deviate=int(len(df_train)*0.25)
for j in range(deviate):
    df_train=df_train.drop(df_train['score'].idxmax())
train_value=df_train['score'].values.tolist()
train_mean=np.mean(train_value)
train_var=np.var(train_value,ddof=1)
train_std=np.std(train_value,ddof=1)
print("train mean=",train_mean,"train variance=",train_var)
print("OK_mean: ",train_mean)
print("OK_var: ",train_var)
print("OK_std: ",train_std)

############ cat 為閥值 也就是分類OK與NG 可更改 ###############
cat=int(train_mean+(train_std)*1) 
###########################
    
########### use train score to set gate ############
ng_conutss=0
ok_countss=0
for g in range(len(test_label)):
    if test_label[g]==1:
        ng_conutss+=1
    else:
        ok_countss+=1
## matplot view
# plt.figure(1, figsize=(3, 3))
# plt.title('query image')
# plt.imshow(qurey.reshape(300,300), cmap=plt.cm.gray)

# print("anomaly score : ", score)
# plt.figure(2, figsize=(3, 3))
# plt.title('generated similar image')
# plt.imshow(pred.reshape(300,300), cmap=plt.cm.gray)

# plt.figure(3, figsize=(3, 3))
# plt.title('anomaly detection')
# plt.imshow(cv2.cvtColor(diff,cv2.COLOR_BGR2RGB))
# plt.show()
scorelist= []
testNG_list = []
testall_list=[]
for i in range(len(X_test_original)):
    # start = cv2.getTickCount()
    score, qurey, pred, diff = anomaly_detection(X_test_original[i])    
    scorelist.append(score)
    if cat <= score:
        print("This is NG sample")
        testNG_list.append(1)
        testall_list.append(1)
        # time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        print ('%d label, %s : done'%(test_label[i], "it is NG"), 'score: %.2f '%score)#, '%.2fms'%time)
        # cv2.imwrite('./fake_img/0729_ori/is_ng_gqurey{}.png'.format(count) , qurey)
    else:
        testall_list.append(0)
    count = str(i)
    #cv2.imwrite('./fake_img/0818/gqurey{}.png'.format(count) , qurey)
    # 儲存位置
    cv2.imwrite('./fake_img/0819/pred{}.png'.format(count), pred)
    cv2.imwrite('./fake_img/0819/diff{}.png'.format(count), diff)
    polt_re(qurey,pred,diff,count)
    print("NG 數量有 %d 個 , OK 數量有 %d 個"%(len(testNG_list),(len(testall_list)-len(testNG_list))))
    print("真實NG 數量有 %d 個 , 真實OK 數量有 %d 個"%(ng_conutss,ok_countss))
#print("測試NG集分數: ",scorelist)


name_list = scorelist
height_list = test_label
df = pd.DataFrame((zip(name_list, height_list)), columns = ['score', 'labe'])
print("test",df)
df.to_csv("./anogan_0818_WT.csv")
### 4. tsne feature view




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    fig=plt.figure()
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
    # 儲存位置
    fig.savefig('./confuse_WT0819.png',dpi=fig.dpi)
    print("confuse save")
    plt.show()

from sklearn.metrics import confusion_matrix, roc_curve, auc
#print(metrics.classification_report(y_test_label, prediction))

cnf_matrix01 = metrics.confusion_matrix(test_label,testall_list)
print("裁減之混淆矩陣 ",cnf_matrix01 )
cnf_matrix = metrics.confusion_matrix(test_label,testall_list)#y_test_label
print("合併後混淆矩陣", cnf_matrix)
target_names = ['OK', 'NG']
# plot_confusion_matrix(predict_y)
plot_confusion_matrix(cnf_matrix, classes=target_names)
# Calculate the sensitivity and specificity
TP = metrics.confusion_matrix(test_label,testall_list, labels=[1, 0])[0, 0]
FP = metrics.confusion_matrix(test_label,testall_list, labels=[1, 0])[1, 0]
FN = metrics.confusion_matrix(test_label,testall_list, labels=[1, 0])[0, 1]
TN = metrics.confusion_matrix(test_label,testall_list, labels=[1, 0])[1, 1]
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
def plt_auc(y_test_label,predict_y):
    fig=plt.figure()
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
    # 儲存位置
    fig.savefig('./AUC_Ganomaly_WT0819.png',dpi=fig.dpi)
plt_auc(test_label,testall_list)


'''
### t-SNE embedding 
### generating anomaly image for test (radom noise image)

from sklearn.manifold import TSNE

random_image = np.random.uniform(0, 1, (100, 300, 300, 1)) #(100, 300, 300, 1)
#random_image = np.random.uniform(0, 1, (100, 28, 28, 1))
print("random noise image")
plt.figure(4, figsize=(2, 2))
plt.title('random noise image')
plt.imshow(random_image[0].reshape(300,300), cmap=plt.cm.gray) #.reshape(300,300)
#plt.imshow(random_image[0].reshape(28,28), cmap=plt.cm.gray)
# intermidieate output of discriminator
model = anogan.feature_extractor()
feature_map_of_random = model.predict(random_image, verbose=1)
feature_map_of_minist = model.predict(X_test_original[:300], verbose=1)
feature_map_of_minist_1 = model.predict(X_test[:100], verbose=1)

# t-SNE for visulization
output = np.concatenate((feature_map_of_random, feature_map_of_minist, feature_map_of_minist_1))
output = output.reshape(output.shape[0], -1)
anomaly_flag = np.array([1]*100+ [0]*300)

X_embedded = TSNE(n_components=2).fit_transform(output)
plt.figure(5)
plt.title("t-SNE embedding on the feature representation")
plt.scatter(X_embedded[:100,0], X_embedded[:100,1], label='random noise(anomaly)')
plt.scatter(X_embedded[100:300,0], X_embedded[100:300,1], label='mnist(anomaly)')
plt.scatter(X_embedded[300:,0], X_embedded[300:,1], label='mnist(normal)')
plt.legend()
plt.show()
'''


#dist.main_diff(qurey,pred)###diff 
##### 如果使用三等份 #####################
def polt_re2(in1,in2,in3,count,testall3,true_lab,z):
    fig=plt.figure()
    plt.subplot(1,3,1)
    plt.title('ori_lab{}'.format(str( true_lab)))
    plt.imshow(in1, cmap=plt.cm.gray)# .reshape(224,224)
    plt.subplot(1,3,2)
    plt.title('simlab{}'.format(str(testall3)))
    plt.imshow(in2, cmap=plt.cm.gray) #.reshape(224,224)
    plt.subplot(1,3,3)
    plt.title('in3 image')
    plt.imshow(in3, cmap=plt.cm.gray)
    # 儲存位置
    fig.savefig('./fake_img/0819/{}_concat_{}.png'.format(z,count),dpi=fig.dpi)
    print("ori concat save")
j=0
count=0
z=[]
conNG_list=[]
contall_list=[]
true_lab_concat=[]
while (j+3 <=len(test_label)):
    testall3=testall_list[j]+testall_list[j+1]+testall_list[j+2]
    true_lab=test_label[j]+test_label[j+1]+test_label[j+2]
    img1 = X_test_original[j]
    img2 = X_test_original[j+1]
    img3 = X_test_original[j+2]
    ori = cv2.vconcat([img1,img2,img3])
    img4 = sim_list[j]
    img5 = sim_list[j+1]
    img6 = sim_list[j+2]
    sim = cv2.vconcat([img4,img5,img6])
    img7 = diff_list[j]
    img8 = diff_list[j+1]
    img9 = diff_list[j+2]
    diff_3=cv2.vconcat([img7,img8,img9])
    if testall3>=1:
        testall3=1
        contall_list.append(1)
        conNG_list.append(testall3)
    else:
        contall_list.append(0)
    if true_lab>=1:
        true_lab=1
        true_lab_concat.append(1)
    else:
        true_lab_concat.append(0)
    if testall3==true_lab:
        z.append("True")
        
    else :
        print("testall3: ",testall3," true_lab",true_lab)
        ori=(ori*255.0).astype(np.uint8)
        # 儲存位置
        cv2.imwrite('./fake_img/0819/{}_diss_{}.png'.format(testall3,count),ori)
        z.append("False")
     
    polt_re2(in1=ori,in2=sim,in3=diff_3,count=count,testall3=testall3,true_lab=true_lab,z=z[count])
    # allscore=score[j]+score[j+1]+score[j+2]
    j=j+3
    count+=1
    print("NG 數量有 %d 個 , OK 數量有 %d 個"%(len(conNG_list),(len(contall_list)-len(conNG_list))))
    print("真實NG 數量有 %d 個 , 真實OK 數量有 %d 個"%(10,10))



cnf_matrix = metrics.confusion_matrix(true_lab_concat,contall_list)#y_test_label
print("合併後混淆矩陣", cnf_matrix)
target_names = ['OK', 'NG']
# plot_confusion_matrix(predict_y)
plot_confusion_matrix(cnf_matrix, classes=target_names)
# Calculate the sensitivity and specificity
TP = metrics.confusion_matrix(true_lab_concat,contall_list, labels=[1, 0])[0, 0]
FP = metrics.confusion_matrix(true_lab_concat,contall_list, labels=[1, 0])[1, 0]
FN = metrics.confusion_matrix(true_lab_concat,contall_list, labels=[1, 0])[0, 1]
TN = metrics.confusion_matrix(true_lab_concat,contall_list, labels=[1, 0])[1, 1]
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

def plt_auc(y_test_label,predict_y):
    fig=plt.figure()
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
    # 儲存位置
    fig.savefig('.AUC_anoganWT0819_.png',dpi=fig.dpi)
try:
    plt_auc(true_lab_concat,contall_list)
except:
    pass
