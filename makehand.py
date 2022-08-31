import cv2 
import os 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


D = "D:\\harden\\dataset\\T85N3 RB\\NG\\"
F= "D:\\harden\\dataset\\T85N3 RB\\NG3\\"
def get_patches(img_arr, size=256, stride=256):
    """
    Takes single image or array of images and returns
    crops using sliding window method.
    If stride < size it will do overlapping.
    
    Args:
        img_arr (numpy.ndarray): [description]
        size (int, optional): [description]. Defaults to 256.
        stride (int, optional): [description]. Defaults to 256.
    
    Raises:
        ValueError: [description]
        ValueError: [description]
    
    Returns:
        numpy.ndarray: [description]
    """    
    # check size and stride
    if size % stride != 0:
        raise ValueError("size % stride must be equal 0")

    patches_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1

    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride - overlapping

        for i in range(i_max):
            for j in range(i_max):
                # print(i*stride, i*stride+size)
                # print(j*stride, j*stride+size)
                patches_list.append(
                    img_arr[
                        i * stride : i * stride + size,
                        j * stride : j * stride + size
                    ]
                )

    elif img_arr.ndim == 4:
        i_max = img_arr.shape[1] // stride - overlapping
        for im in img_arr:
            for i in range(i_max):
                for j in range(i_max):
                    # print(i*stride, i*stride+size)
                    # print(j*stride, j*stride+size)
                    patches_list.append(
                        im[
                            i * stride : i * stride + size,
                            j * stride : j * stride + size,
                        ]
                    )

    else:
        raise ValueError("img_arr.ndim must be equal 3 or 4")

    return np.stack(patches_list)

vou=0
pc=0
for i in tqdm(os.listdir(D)): 
    img1 = cv2.imread(D+i)
    h,w=img1.shape[0],img1.shape[1]
    img1 = img1[75:650,1000:1900] 
    #print(h,w) 
    xx=get_patches(img_arr=img1, size=516, stride=516)
    
    for k in range(len(xx)):
        x3=xx[k]
        cv2.imwrite(F+"%s_ngb.jpg" %str(pc), x3)
        pc+=1
    

    #img2 = img1[0:360,0:640]
    #cv2.imwrite(F+"%s_nga.jpg" %str(vou), img2) 
    #img3 = img1[361:722,641:1281] 
    #cv2.imwrite(F+"%s_ngb.jpg" %str(vou), img3 ) 
    #img4 = img1[723:,1281:]
    #cv2.imwrite(F+"%s_ngc.jpg" %str(vou), img4 ) 
    

    vou +=1
    if vou >=500:
        break


    
#cv2.imshow("ori",img)
#cv2.imshow("img2",img2)
#img_name="D:\\harden\\dataset\\CF_Marco\\CF_cut_ok\\0.jpg"
#cv2.imwrite(img_name, img2) 
#cv2.waitKey(0)
#cv2.destroyAllWindows()