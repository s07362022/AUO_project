
import load_dataset as ld 
import cv2
import matplotlib.pyplot as plt
import numpy as np
'''
D1 = "D:\\harden\\dataset\\H1_water_de\\OK2\\"
E = "D:\\harden\\dataset\\H1_water_de\\water_cut_NG\\"
#F =  "D:\\harden\\dataset\\CF_Marco\\CF_cut_ok\\"
#B = "D:\\harden\\dataset\\CF_Marco\\CF_cut_ng\\"
test_ldy=[]
train_ldx=[]
train_ldy=[]
ld.d(D1,train_ldx,test_ldy)
ld.d(E,test_ldx,test_ldy,1)
#ld.d(F,test_ldx,test_ldy,2)
#ld.d(B,test_ldx,test_ldy,3)

test_ldx=np.array(test_ldx)

test_ldx=(test_ldx).astype(np.uint8)#*255


#plt.imshow(test_ldx[0])
#plt.show()
'''
# 均方相似度


def getss(list):				     #計算方差
    avg=sum(list)/len(list)          #計算平均值
    ss=0
    for l in list:                   #計算方差
        ss+=(l-avg)*(l-avg)/len(list)
    return ss


def getdiff(img):                     #獲取每行像素平均值
    Sidelength=224                     #定義邊長
    img=cv2.resize(img,(Sidelength,Sidelength),interpolation=cv2.INTER_CUBIC)
    #if img.shape[2] ==None :
        #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=img
    avglist=[]                        #avglist列表保存每行像素平均值
    for i in range(Sidelength):       #計算每行均值，保存到avglist列表
        avg=sum(gray[i])/len(gray[i])
        avglist.append(avg)
    return avglist
#train_mse = []
#test_mse = []
def main_diff(x1,x2):
#讀取測試圖片
    img1=x1
    diff1=getdiff(img1)
    mse1=getss(diff1)
    #print('img1 方均差:',mse1)
    #train_mse.append(mse1)

    #讀取測試圖片
    img11=x2
    diff11=getdiff(img11)
    mse2=getss(diff11)
    #print('img11 方均差:',mse2)
    #test_mse.append(mse2)

    x=range(224)

    plt.figure("avg")
    plt.plot(x,diff1,marker="*",label="$train$")
    plt.plot(x,diff11,marker="*",label="$test$")
    plt.title("avg")
    plt.legend()
    #plt.show()
    return mse1,mse2
  
# for k in range(len(test_ldx)):
    # main_diff(train_ldx[k].astype(np.uint8),test_ldx[k].astype(np.uint8))