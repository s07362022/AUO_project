import cv2
import os 
IMAGE_SIZE = 512
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #get size
    h, w , _= image.shape
    
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
W="D:\\harden\\dataset\\H1_water_de\\w\\"
def d(D,images,labels,dir_counts = 0,vou=0):
    vou=0
    for i in os.listdir(D):
        # try:
        img1 = cv2.imread(D+i)
        print(D+i)
        #cv2.imwrite("D:\\harden\\dataset\\H1_water_de\\half_resizei\\oring_{}.jpg".format(str(vou)),img1)
        #print(img1.shape)
        #img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))/255.0
        #img1=np.expand_dims(img1,axis=-1)
        #if dir_counts in [0,1]:
        img1=cv2.resize(img1,(IMAGE_SIZE, IMAGE_SIZE)) #/ 255.0
        #img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
        #cv2.imwrite("D:\\harden\\unet2\\keras-Unet-master\\data\\raw\\images\\img_{}.jpg".format(str(vou)),img1)
        cv2.imwrite(D+i,img1)
        #print(dir_counts)
        #else:
            #img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE) #/255.0
            #print("resize back",dir_counts)
            #cv2.imwrite(W+"%s.tiff" %str(vou), img1)
        #img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE) 
        #print(img1.shape)
        img1 = img1 #/255.0
        images.append(img1)
        labels.append(dir_counts)
        label = dir_counts
            
        #except:
            #print("error")
        vou +=1
        #if vou >=2:
            #break
    print("A already read")
    return(images,labels)

ok_ptah='D:\\harden\\unet2\\keras-Unet-master\\data\\raw\\labels\\'
ori_ok=[]
re_ok=[]
d(ok_ptah,ori_ok,re_ok)