import os

#資料夾的檔案
all_files = []
all_paths = []
#count=0
for path, dir, file in os.walk("D:\\harden\\unet\\unet-tf2-main\\VOCdevkit\\VOC2007\\JPEGImages\\"): #資料夾路徑
    for f in file:
        if os.path.splitext(f)[-1] in ['.png']:
            all_files.append(f)
            all_paths.append(path)
            new_fname =os.path.splitext(f)[0]+".jpg"  #檔案名稱 'img_%d'%(count)+".png"
            os.rename(os.path.join(path, f), os.path.join(path, new_fname))
            #count+=1
#for i in range(len(all_files)):            
    #print(all_paths[i] + "\t" + all_files[i])

#for i in range(len(all_files)):            
    #print(all_paths[i]  + all_files[i])
    #k =0 #檔案名稱從1開始
    #fname = all_files[i]
    #new_fname = os.path.splitext(f)[0]+".jpg"  #檔案名稱
    #os.rename(os.path.join(all_paths[i], fname), os.path.join(all_paths[i], new_fname))