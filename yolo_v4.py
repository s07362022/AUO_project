import cv2
from matplotlib.pyplot import box
import numpy as np
import os
import shutil

#PVmv
def initNet():
    CONFIG = 'yolov4-tiny-chipping-default-anchor.cfg'
    WEIGHT = 'yolov4-tiny-chipping-default-anchor_final.weights'
    # WEIGHT = './train_finished/yolov4-tiny-myobj_last.weights'
    net = cv2.dnn.readNet(CONFIG, WEIGHT)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255.0)
    model.setInputSwapRB(True)
    # print(model)
    return model

#
def nnProcess(image, model):
    classes, confs, boxes = model.detect(image, 0.1, 0.1)
    return classes, confs, boxes

#A
def drawBox(image, classes, confs, boxes):
    new_image = image.copy()
    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box
        if x - 18 < 0:
            x = 18
        if y - 18 < 0:
            y = 18
        cv2.rectangle(new_image, (x - 18, y - 18), (x + w + 20, y + h + 24), (0, 255, 0), 3)
    return new_image

# 
def cut_img(image, classes, confs, boxes):
    cut_img_list = []
    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box
        if x - 31 < 0:
            x = 31
        if y - 40 < 0:
            y = 41
        #cut_img = image[y - 30:y + h + 30, x - 18:x + w + 25]
        cut_img = image[y - 120: y + h + 90, x - 10: x + w + 10]
        cut_img_list.append(cut_img)
    return cut_img_list[0]

# xsweBz(|)
def saveClassify(image, output):
    cv2.imencode(ext='.tiff', img=image)[1].tofile(output)

if __name__ == '__main__':
    source = 'D:\\harden\\dataset\\test02\\'
    savepath01='D:\\harden\\dataset\\get\\'
    # source = './public_training_data/public_testing_data/'
    files = os.listdir(source)
    print(' @ {} i'.format(len(files)))
    print(' }lYOLOV4...')
    model = initNet()
    success = fail = uptwo = 0
    number = 1
    crypt_num = 0
    for file in files:
        print('  {}i'.format(number)," W: {}".format(file))
        #img = cv2.imdecode(np.fromfile(source+file, dtype=np.uint8), -1)
        #classes, confs, boxes = nnProcess(img, model)
        try :
            img = cv2.imdecode(np.fromfile(source+file, dtype=np.uint8), -1)
            #print(img)
            #img = cv2.resize(img,(700,700))   ### resize
            classes, confs, boxes = nnProcess(img, model)
            #print(classes)
            # print(len(boxes))
            if len(boxes) == 0:
                # xsl
                # saveClassify(img, './public_training_data/YOLOV4_pre/fail/' + file)
                saveClassify(img, savepath01 + file)
                fail += 1
                print('  G{}'.format(file))
                # cv2.imshow('img', img)
            # elif len(boxes) >= 2:
                # print('  WL2')
                # box_img = drawBox(img, classes, confs, boxes)
                # print(classes[0][0])
                # if classes[0][0] == 0:
                    # cut = cut_img(img, classes, confs, boxes)
                    # saveClassify(cut, 'E:\\workspace\\project_\\smoke_0407_img\\yolo_40\\' + file)
                    # print( "U\" )
                    # success += 1
                    # uptwo += 1
            else:
                # 
                frame = drawBox(img, classes, confs, boxes)
                # 
                #       if classes !=0:
                # cut = cut_img(img, classes, confs, boxes)
                saveClassify(frame, savepath01 + file)
                #print( "SAVE" )
                #cut = cut_img(img, classes, confs, boxes)
                # xs
                #saveClassify(cut, 'E:\\workspace\\project_\\smoke_data_test\\1_28_6_55_cut\\' + file)
                success += 1
                print('  \G{}'.format(file))
                crypt_num +=len(boxes)
                    
                   
                    # success += 1
                # print('=' * 60)
                # cv2.waitKey()
            number += 1
        except:
            pass
    print(' {')
    print(' `pG\ {} iB {} i'.format(success, fail))
    print(' `pGcrypt : {}'.format(crypt_num))
    #print(' WL {} i'.format(uptwo))
        
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()