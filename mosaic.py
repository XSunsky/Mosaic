# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:41:56 2019
     将图片马赛克化拼接

将图片（待拼接）放入images文件夹中；原图放入当前目录
原图压缩比ratio尽量大一些，即效果图dstSize小一些；
小图尺寸minImgSize尽量小一些，不然效果图很大！！！
@author: 56472
"""
import cv2
import numpy as np
import glob
import random
import os
from tqdm import tqdm    # 添加进度条


from numba import autojit   # 加速程序的，但感觉没啥用
@autojit

## 使用该函数可以路径含有中文
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
#    cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

## 建立字典
def get_ImagesDict(ImagesDir, minImgSize):
    ImagesList = glob.glob(ImagesDir)
    ImagesDict = {}
    
    pbar = tqdm(total=100)
    for i in range(len(ImagesList)):
        img = cv_imread(ImagesList[i])
        imgSquare = get_square(img, minImgSize)
        imgMean = int(np.mean((np.mean(imgSquare[:,:,0]), 
                               np.mean(imgSquare[:,:,1]), 
                               np.mean(imgSquare[:,:,2]))))
        
        if imgMean not in ImagesDict:
            ImagesDict.setdefault(str(imgMean), [])
            ImagesDict[str(imgMean)].append(ImagesList[i])
        else:
            ImagesDict[str(imgMean)].append(ImagesList[i])
        
        pbar.update(100/len(ImagesList))
        
    pbar.close() 
    return ImagesDict

## 保存或读取字典
def save_or_load_Dict(ImagesDir, mosaicDict):
    if not os.path.exists(mosaicDict):
        # 建立字典，并保存
        ImagesDict = get_ImagesDict(ImagesDir, 300)
        f = open(mosaicDict,'w')
        f.write(str(ImagesDict))
        f.close()
    else:   
        ## 读取字典
        f = open(mosaicDict,'r')
        a = f.read()
        ImagesDict = eval(a)
        f.close()
    
    return ImagesDict

## 将图片剪裁至固定大小，且为正方形
def get_square(image, minImgSize):
    [height, width, _] = image.shape
    if height == width:
        result = cv2.resize(image, (minImgSize, minImgSize))
        return result
    else:
        zoomratio = minImgSize/min(width, height)
        result = cv2.resize(image, None, fx=zoomratio, fy=zoomratio)

        result = result[0:minImgSize, 0:minImgSize]
        return result

## 找到最近像素值
def find_nearestKey(ImagesDictKeys, key):
    if key in ImagesDictKeys:
        nearestKey = key
    else:
        nearestKey = ImagesDictKeys[(np.abs(ImagesDictKeys-key)).argmin()]
    
    return nearestKey

## 找到合适的贴怕
def findRightImg(ImagesDict, ImagesDictKeys, key, minImgSize):
    nearestKey = find_nearestKey(ImagesDictKeys, key)
    tmpImage = cv_imread(ImagesDict[str(nearestKey)][random.randint(0,len(ImagesDict[str(nearestKey)])-1)])
    tmpImage = get_square(tmpImage, minImgSize)
    
    return tmpImage

## 拼接
def mosaic(srcImage, minImgSize, ImagesDict):
    M, N = srcImage.shape

    ImagesDictKeys = np.array(list(map(int,ImagesDict.keys())))
    
    pbar = tqdm(total=100)
    
    for i in range(M):
        if i == 0:
            for j in range(N):
                if j == 0:
                        tmpImage = findRightImg(ImagesDict, ImagesDictKeys, srcImage[i][j], minImgSize)
                        rowImage = tmpImage
                else:
                        tmpImage = findRightImg(ImagesDict, ImagesDictKeys, srcImage[i][j], minImgSize)
                        rowImage = np.hstack([rowImage, tmpImage])
            dstImage = rowImage
        else:
            for j in range(N):
                if j == 0:
                        tmpImage = findRightImg(ImagesDict, ImagesDictKeys, srcImage[i][j], minImgSize)
                        rowImage = tmpImage
                else:
                        tmpImage = findRightImg(ImagesDict, ImagesDictKeys, srcImage[i][j], minImgSize)
                        rowImage = np.hstack([rowImage, tmpImage])
            colImage = rowImage
        
            dstImage = np.vstack([dstImage, colImage])
        pbar.update(100/M)
            
    pbar.close() 
    
    return dstImage




srcImgDir = '3.jpg'
ImagesDir = 'images/*'
DictName = 'mosaic.txt'
minImgSize = 100
dstSize = 80
#ratio = 0.4

## 读取原图
srcImg = cv2.imread(srcImgDir,0)
ratio = dstSize / max(srcImg.shape)
srcImg = cv2.resize(srcImg, None, fx=ratio, fy=ratio)

## 读取字典
ImagesDict = save_or_load_Dict(ImagesDir, DictName)
print('保存，加载字典完成！')

## 拼接
mosaicImg = mosaic(srcImg, minImgSize, ImagesDict)
print('拼接完成！！')

## 显示或保存
cv2.imwrite('MOSAIC.jpg', mosaicImg)
print('保存完成！！！')
#cv2.waitKey()
#cv2.destroyAllWindows()


