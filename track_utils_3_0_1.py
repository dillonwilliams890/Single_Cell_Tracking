#%%
import numpy as np
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from scipy.optimize import curve_fit
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import time
import trackpy as tp
import math
import scipy.interpolate as interp
from scipy.ndimage import uniform_filter1d
from PIL import Image
# import tqdm
# import einops
# import random
# import pathlib
# import itertools
# import collections
# import keras
# import os
import ruptures as rpt
# from keras import layers
import seaborn as sns
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from PIL import Image
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
import pathlib
# from tensorflow.keras.optimizers import Adam
import ipyplot
import glob
from PIL import Image
from natsort import natsorted
from scipy.ndimage.morphology import binary_dilation
# from CNN_utils import *

#A few fit functions
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def linear(x, m, b):
    y = m * x + b
    return (y)

def exponential(x,P,Yo,k):
    y=P + ( Yo - P ) * np.exp( -k * x)
    return(y)

def piecewise_exponential(x, x0, P, Yo, k):
    return np.piecewise(x, [x < x0, x>=x0], [lambda x:Yo, lambda x:P + ( Yo - P ) * np.exp( -k * (x-x0))])

#Thsin function finds the x location of the cell 
def find_cell_x(frame):
    img=frame.astype('uint8')
    Gy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=13)
    aGy=abs(Gy)
    means=(np.mean(aGy, axis=0))
    pt=[(np.argwhere(means == means.max()))][0][0]
    if pt<40 or pt>320:
        pt=180
    return pt

#This function pads the wall to ensure adequate removal
def pad_wall(grad):
    W = 2 # window length of neighbors
    thresh = 0.1
    mask = grad < thresh
    kernel = np.ones(2*W+1)
    mask_extended = binary_dilation(mask, kernel)
    grad[mask_extended] = 0
    return grad

#This function locates the wall using Sobel Gradients
def find_wall(img):
    img=img.astype('uint8')
    Gy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    aGy=abs(Gy)
    grad=np.median(aGy,axis=1)
    grad=np.where(grad > 10, 0, 1)
    grad=pad_wall(grad)
    return grad

#Normalize the frames suing the bleed through file
def bleed(frame, BL):
    L430430=(BL[0])
    L410410=(BL[1])
    L630B=(BL[2])
    L430630=(BL[3])
    L410630=(BL[4])
    L630630=(BL[5])
    blr=(BL[6])
    B430= frame[0]-blr[0,:,:]
    B410= frame[1]-blr[0,:,:]
    R430= frame[2]-blr[1,:,:]
    R410= frame[3]-blr[1,:,:]

    B430=B430*L430430-B430*L630B
    B410=B410*L410410-R410*L630B
    R430=R430*L630630-B430*L430630
    R410=R410*L630630-B410*L410630

    frameBL=[B430,B410,R430,R410]
    return frameBL

#This function is used to check if a cell found in a frame is close to the cell in the previous frame, to avoid getting different cells. It is not being used at the moment
def euler(f, thresh, x_old, y_old): 
    # print(len(f))
    pt=[0,0,0] 
    if len(f)==1:
        pt=[f.x.values[0],f.y.values[0], f.signal.values[0]]
    elif len(f)==2:
        pt1=[f.x.values[0],f.y.values[0], f.signal.values[0]]
        pt2=[f.x.values[1],f.y.values[1], f.signal.values[1]]
        if pt1[2]>thresh and pt2[2]>thresh:
            dist1 = math.hypot(x_old - pt1[0], y_old - pt1[1])
            dist2 = math.hypot(x_old - pt2[0], y_old - pt2[1])
            if dist1<dist2:
                pt=pt1
            elif dist1>dist2:
                pt=pt2  
        elif pt1[2]>thresh and pt2[2]<thresh:
            pt=pt1
        elif pt1[2]<thresh and pt2[2]>thresh:
            pt=pt2
        if pt1[2]>thresh and pt2[2]>thresh:
            dist1 = math.hypot(x_old - pt1[0], y_old - pt1[1])
            dist2 = math.hypot(x_old - pt2[0], y_old - pt2[1])
            if dist1<dist2:
                pt=pt1
            elif dist1>dist2:
                pt=pt2 
        elif pt1[2]>thresh and pt2[2]<thresh:
            pt=pt1
        elif pt1[2]<thresh and pt2[2]>thresh:
            pt=pt2
    else:
        pt=[0,0,0]   
    # print(pt)
    return pt 

#normalize images
def rescale(img):
    norm_img = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return norm_img
#noralize, invert, and mask blue images
def rescale_blue(img, mask):
    masked = (img)*mask
    masked=1-masked
    masked=rescale(masked)
    masked = (masked)*mask
    return masked

#nomalize and mask red images
def rescale_red(img, mask):
    masked = (img)*mask
    masked=rescale(masked)
    masked = (masked)*mask
    return masked

#Calcuate the absorption of hemoglobin 
def mass(Hb):
    parea=(6.9/20)**2; 
    Hb=Hb.astype('uint8')
    Gy = cv.Sobel(Hb, cv.CV_64F, 0, 1, ksize=3)
    aGy=abs(Gy)
    means=(np.mean(aGy, axis=0))
    x=np.delete(Hb,np.where(means>20), axis=1)
    means=(np.mean(x, axis=1))
    Hbnorm=Hb/means[:,None]
    Hbnorm[Hbnorm <= 0] = 1
    Hbnorm[Hbnorm >= 1] = 1
    hbmass=((parea*(10**-8)*64500*np.sum(np.sum((-np.log10(Hbnorm))))))
    
    return hbmass

#create a mask for the cell and normalize and mask the images
def masking(img):
    cx=0;cy=0
    img=img.astype('uint8')
    cell_mask = np.zeros_like(img)
    masked = np.zeros_like(img)
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU) 
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
            # Calculate area and remove small elements
            area = cv.contourArea(cnt)  
            hull = cv.convexHull(cnt)
            perimeter = cv.arcLength(hull, True)  
            # x, y, w, h = cv.boundingRect(cnt)   
            area = cv.contourArea(hull, True)
            circ = 2*np.sqrt(np.pi*area)/perimeter
            if area > 50 and area < 1000 and circ>0.7:
                cell_mask = np.zeros_like(img)
                masked = np.zeros_like(img)
                M = cv.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                
                cv.drawContours(cell_mask, [hull], -1,(1), -1)
                img=rescale(img)
                cell=(img)
                cell_mask=(cell_mask)
                masked=rescale_blue(cell,cell_mask)
                
    if masked.shape!=(20,40) or cell_mask.shape!=(20,40):
            masked=np.zeros((20,40))
            cell_mask=np.zeros((20,40))
    
    return masked, cell_mask, cx, cy

#calulcate the volume on the cell
def volume(cell, mask, top):
    channel=11
    # x=pt[0]; y=pt[1]
    # cell=(img[int(y-15):int(y+15), int(x-20):int(x+20)])
    if len(mask[0])<40:
        vol=np.nan
    else:
        kernel = np.ones((5,5),np.uint8)
        dilation = cv.dilate(mask,kernel,iterations = 1)
        masked = (cell*(mask))
        sur=np.mean(masked)
        parea=(6.9/20)**2
        nimg=cell*(dilation)/sur
        nimg[nimg <= 0] = 0.01
        transmission = sur/top
        alf = -np.log(transmission)/channel
        tcell=np.log(nimg)/alf
        vol=np.sum(tcell*mask)*parea
    return vol
   
#Calcuate oxy and deoxy hemoglobin mass and cell saturation
def saturation(frames): 
     #camera pixel area
    b=[frames[0],frames[1]]
    #Molecular absorbtion coefficients
    w430_o = 2.1486*(10**8)
    w430_d = 5.2448*(10**8)
    w410_o = 4.6723*(10**8)
    w410_d = 3.1558*(10**8)

    mass410=mass(b[1])
    mass430=mass(b[0])

    e=mass410 #410
    f=mass430 #430
    #Set absorbtion values to equation constants
    a=w410_d
    b=w410_o
    c=w430_d
    d=w430_o
                
    #Calcuate mass of oxygenated and deoxygenated hemoglobin
    Mo=(e*c-a*f)/(b*c-a*d)
    Md=(b*f-e*d)/(b*c-a*d)

    saturation = Mo/(Mo+Md)
    hbmass=Mo+Md
    # print(Mo)
    # print('$')
    # print(Md)
    return saturation, hbmass

#resize the images, and process them for the neural net, not being used at the moment
def net(frames, top):

    B430 = frames[0]
    B410 = frames[1]
    imgR430 = frames[2]
    imgR410 = frames[3]

    imgB430, mask430, cx430, cy430=masking(B430)
    imgB410, mask410, cx410, cy410=masking(B410)

    mask430[mask430<0] = 0
    mask410[mask410<0] = 0
    imgB410[imgB410<0] = 0
    imgB430[imgB430<0] = 0
    imgR410[imgR410<0] = 0
    imgR430[imgR430<0] = 0


    vol430=volume(imgR430, mask430, top[0])
    vol410=volume(imgR410, mask410, top[1])
    ## todo get cx and make square

    imgR410=rescale(imgR410)
    imgR410=rescale_red(imgR410,mask410)
    imgR430=rescale(imgR430)
    imgR430=rescale_red(imgR430,mask430)

    imgB430=np.pad(imgB430, 10, mode='constant')
    imgB410=np.pad(imgB410, 10, mode='constant')
    imgR430=np.pad(imgR430, 10, mode='constant')
    imgR410=np.pad(imgR410, 10, mode='constant')
    

    imgB430=(imgB430[int(cy430):int(cy430+20), int(cx430):int(cx430+20)])
    imgB410=(imgB410[int(cy410):int(cy410+20), int(cx410):int(cx410+20)])
    imgR430=(imgR430[int(cy430):int(cy430+20), int(cx430):int(cx430+20)])
    imgR410=(imgR410[int(cy410):int(cy410+20), int(cx410):int(cx410+20)])

    imgR=imgR430/2+imgR410/2

    imgB430=cv.resize(imgB430, dsize=(81, 81), interpolation=cv.INTER_LINEAR)
    imgB410=cv.resize(imgB410, dsize=(81, 81), interpolation=cv.INTER_LINEAR)
    imgR=cv.resize(imgR, dsize=(81, 81), interpolation=cv.INTER_LINEAR)
    img = np.zeros([81,81,3])
    img[:,:,0] = imgB430
    img[:,:,1] = imgB410
    img[:,:,2] = imgR
    # img[:,:,2] = (imgB430+imgB410)/2
    
    vol=(vol430+vol410)/2
    # print(vol)
    return img, vol

#Seperate the RGB images and create a cell ROI in each
def segment(frames,x_old, y_old,BL):
    img=frames
    if x_old<1 or y_old<1:   
        x_old=180; y_old=135
    L1=img[0][1::2, 1::2]
    L2=img[1][1::2, 1::2]
    R1=img[0][::2, ::2]
    R2=img[1][::2, ::2]
    if np.mean(L1)>np.mean(L2):
        b=[L1,L2]
        frame=[L1,L2,R1,R2]
    else:
        b=[L2,L1]
        frame=[L2,L1,R2,R1]
    frame=bleed(frame,BL)
    top=[np.mean(frame[2][int(0):int(10), int(0):int(40)]),np.mean(frame[3][int(0):int(10), int(0):int(40)])]
    # plt.imshow(frame[0])
    x430=find_cell_x(frame[0])
    x410=find_cell_x(frame[1])
    frame=[frame[0][:,int(x430-40):int(x430+40)],frame[1][:,int(x410-40):int(x410+40)],frame[2][:,int(x430-40):int(x430+40)],frame[3][:,int(x410-40):int(x410+40)]]
    
    grad430=find_wall(frame[2])
    grad410=find_wall(frame[3])
    b430=np.delete(frame[0], np.where(grad430 < 1), axis=0)
    # plt.imshow(b430)
    r430=np.delete(frame[2], np.where(grad430 < 1), axis=0)
    pt430=[(np.argwhere(b430 == b430.min()))][0][0]
    y430=pt430[0]; x430=pt430[1]
    b410=np.delete(frame[1], np.where(grad410 < 1), axis=0)
    r410=np.delete(frame[3], np.where(grad410 < 1), axis=0)
    pt410=[(np.argwhere(b410 == b410.min()))][0][0]
    y410=pt410[0]; x410=pt410[1]
    frames=[b430[int(y430)-10:int(y430)+10, int(x410-20):int(x410+20)],b410[int(y410)-10:int(y410)+10, int(x410-20):int(x410+20)],r430[int(y430)-10:int(y430)+10, int(x430-20):int(x430+20)],r410[int(y410)-10:int(y410)+10, int(x410-20):int(x410+20)]]

    return frames, top, pt430, pt410

#Load the BL file
def getBL(): 
    with h5py.File('BL.h5', 'r') as BL:
        L430430=(BL['L430430'][:])
        L410410=(BL['L410410'][:])
        L630B=(BL['L630B'][:])
        L430630=(BL['L430630'][:])
        L410630=(BL['L410630'][:])
        L630630=(BL['L630630'][:])
        blr=(BL['blr'][:])
    BL=[L430430,L410410,L630B,L430630,L410630,L630630,blr]
    return BL

 #Run the anaylsis in the video
def main_run(video):
    BL=getBL()
    i=0
    x_old=0
    y_old=0
    saturations=[]
    imgs=[]
    volumes=[]
    hgb=[]
    x=[0,0,0]
    y=[0,0,0]
    I430=[]
    I410=[]
    while i<len(video):
        frame1=video[i][0:2]
        frame2=video[i][2:4]
        frame3=video[i][2:6]
        frame4=video[i][6:8]
        frames=[frame1,frame2, frame3, frame4]
        for frame in frames:
            x_old=np.mean(x[-3:])
            y_old=np.mean(y[-3:])
            frame, top, pt430, pt410 = segment(frame,x_old, y_old,BL)
            # print(pt430)
            # print(pt410)
            if pt430[1]>=21 and pt410[1]>21 and pt430[1]<339 and pt410[0]<339 and pt430[0]>11 and pt410[0]>11 and pt430[0]<239 and pt410[0]<239:
                sats, hbmass=saturation(frame)
                saturations.append(sats)
                hgb.append(hbmass)
                # img, vol=net(frames, top)
                # imgs.append(img)
                # volumes.append(vol)
                x.append(pt430[1])
                # y.append(pt410[1])
                # I430.append(blue_top[0])
                # I410.append(blue_top[1])
            else:
                saturations.append(np.nan)
                # volumes.append(np.nan)
                hgb.append(np.nan)
                # imgs.append(np.zeros([81,81,3]))
                x.append(300)
                # I430.append(np.nan)
                # I410.append(np.nan)
        i=i+1
    return imgs, saturations, volumes, x, hgb

#view the video
def veiw(video):
    BL=getBL()
    i=0
    x_old=0
    y_old=0
    x=[0,0,0]
    y=[0,0,0]
    while i<len(video):
        frame1=video[i][0:2]
        frame2=video[i][2:4]
        frame3=video[i][2:6]
        frame4=video[i][6:8]
        frames=[frame1,frame2, frame3, frame4]
        for frame in frames:
            # frame = video[i]#np.maximum(data2[i][0],data2[i][1])
            x_old=np.mean(x[-3:])
            y_old=np.mean(y[-3:])
            frames, top, pt430, pt410 = segment(frame,x_old, y_old,BL)
            gray=cv.resize(frame[0], dsize=(720, 540), interpolation=cv.INTER_CUBIC).astype("uint8")
            # print(pt430)
            # gray = cv.cvtColor(frame[0], cv.COLOR_BayerBG2GRAY)
            if pt430[1]>21 and pt410[1]>21 and pt430[1]<339 and pt410[0]<339 and pt430[0]>11 and pt410[0]>11 and pt430[0]<239 and pt410[0]<239:
                # frames, top, b=crop(frame, pt430, pt410)
                sat, hb=saturation(frames)
                pt430= np.asarray(pt430, dtype=np.float32)
                pt410= np.asarray(pt410, dtype=np.float32)
                ptres430=2*pt430; ptres410=2*pt410
                x.append(pt410[1]); y.append(pt410[0])
                locx=int(ptres410[1]); locy=int(ptres410[0])
                # img = gray[y-40:y+40,x-40:x+40]
                cv.rectangle(gray, (locx-40, locy-20), (locx+40, locy+20), (255,100,200),2)
                # img=cv.resize(img, dsize=(200, 200), interpolation=cv.INTER_CUBIC).astype("uint8")
                cv.putText(gray, str('%.2f' %sat), (locx-45,locy-45), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                cv.putText(gray, str('%f' %i), (45,45), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            cv.imshow('Single Track', gray)
                # # # press 'q' to break loop and close window
            cv.waitKey(5)
        i=i+1
    # print(xf)
    cv.destroyAllWindows()

#write the video to .avi format
def write(video, name):
    BL=getBL()
    i=0
    x_old=0
    y_old=0
    x=[0,0,0]
    y=[0,0,0]
    vid = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'), 20, (720, 540), False)
    while i<len(video):
        frame = video[i]
        x_old=np.mean(x[-3:])
        y_old=np.mean(y[-3:])
        pt430, pt410, frame = segment(frame,x_old, y_old,BL)
        gray=cv.resize(frame[0], dsize=(720, 540), interpolation=cv.INTER_CUBIC).astype("uint8")
        # gray = cv.cvtColor(frame[0], cv.COLOR_BayerBG2GRAY)
        if pt430[0]>21 and pt410[0]>21 and pt430[0]<349 and pt410[0]<349 and pt430[1]>11 and pt410[1]>11 and pt430[1]<239 and pt410[1]<239:
            frames, top=crop(frame, pt430, pt410)
            sat, hb=saturation(frames)
            pt430= np.asarray(pt430, dtype=np.float32)
            pt410= np.asarray(pt410, dtype=np.float32)
            ptres430=2*pt430; ptres410=2*pt410
            # img = gray[y-40:y+40,x-40:x+40]
            # cv.rectangle(gray, (locx-20, locy-20), (locx+20, locy+20), (255,100,200),2)
            # # img=cv.resize(img, dsize=(200, 200), interpolation=cv.INTER_CUBIC).astype("uint8")
            cv.putText(gray, str('%f' %i), (45,45), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        vid.write(gray)
        i=i+1
    # print(xf)
    cv.destroyAllWindows()
    vid.release()
    
#Chunk the video into N frame chunks for the CNN
def chunk(video, num):
    BL=getBL()
    i=0
    stack=[]
    x_old=0
    y_old=0
    x=[0,0,0]
    y=[0,0,0]
    m,n,o,p = video.shape[:]
    data_new = np.rollaxis(video,0,1).reshape(-1,2,o,p)
    mod=len(data_new)%num
    length=(len(data_new)-mod)/num
    chunked=np.array_split(data_new, length)
    while i<length:
        j=0
        clip=[]
        frames = chunked[i]
        while j<num:
            frame=frames[j]
            x_old=np.mean(x[-3:])
            y_old=np.mean(y[-3:])
            pt430, pt410, img = segment(frame,x_old, y_old,BL)
            if pt430[0]>21 and pt410[0]>21 and pt430[0]<339 and pt410[0]<339 and pt430[1]>11 and pt410[1]>11 and pt430[1]<239 and pt410[1]<239:
                cell, top=crop(img,pt430,pt410)
                x.append(pt410[0])
                y.append(pt410[1])
                cell_img, vol=net(cell, top)
                clip.append(cell_img)
            j=j+1    
        stack.append(clip)    
        i=i+1
    return stack

#Save the chunk for CNN anaylsis
def save_stack(stack, name):
    i=0
    imgs=[]
    # vid = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'), 10, (81, 81), True)
    # while i<19:
    while i<len(stack):
        img = cv.normalize(stack[i], None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype("uint8")
        
        if np.max(img[:,:,0])>0 and np.max(img[:,:,1])>0 and np.max(img[:,:,2])>0 and np.min(img[:,:,0])==0 and np.min(img[:,:,1])==0 and np.min(img[:,:,2])==0:
        # img=cv.resize(norm, dsize=(100, 100), interpolation=cv.INTER_CUBIC).astype("uint8")
            imgs.append(img)
        i=i+1
    if len(imgs)==len(stack):
        j=0
        vid = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'), 10, (81, 81), True)
        while j<len(imgs):
            vid.write(imgs[j])
            j=j+1
        vid.release() 

def partition(alist, indices):
    return [alist[i:j] for i, j in zip([0]+indices, indices+[None])]

#Check if a 
def vol_step(volumes):
    double=False
    jump=[]
    vol_dif=np.asarray(volumes[50:])
    algo = rpt.Pelt(model="rbf").fit(vol_dif)  # Use the Pelt algorithm with L2 norm
    result = algo.predict(pen=10)  # Penalty for each change point
    i=0
    parts=partition(vol_dif,result[:-1])
    if len(parts)>1:
        while i<len(parts)-1:
            jump.append(abs((np.mean(parts[i])-np.mean(parts[i+1]))))
            i=i+1
    else:
        jump=0
    if np.max(jump)>10:
        double=True
    return double

def CNN(subset_paths,model): #Run the CNN and display the resutls
    # new_model=keras.models.load_model('CNN/model_time_2.keras') #load the model
    n_frames = 10
    batch_size = 8

    output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                        tf.TensorSpec(shape = (), dtype = tf.int16))

    tmp_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['tmp'], n_frames),
                                            output_signature = output_signature,)

    tmp_ds = tmp_ds.batch(batch_size)
    predicted = model.predict(tmp_ds)
    predicted = tf.concat(predicted, axis=0)
    # predicted = tf.argmax(predicted, axis=1)
    
    
    # ipyplot.plot_images(sorted_cells, sorted_preds,max_images=500, img_width=50)
    return predicted   

###ANAYLSIS###

def batch_analyze(file_path):
    cells = {}
    cell_paths=[]
    raw=[]
    # model=keras.models.load_model('CNN/model_time_3.keras') #load the model
    files = glob.glob(file_path, 
                    recursive = True)
    files=natsorted(files)
    n=0
    for file in files:
        print(file)
        with h5py.File(file, 'r') as hf:
            video = hf['data'][:]
        _, saturations, volumes, x, hbmass = main_run(video)
        saturations=np.asarray(saturations)
        df_name='df0_'+str(n)
        cell_paths.append(file)
        dict = {'sat': saturations, 'mass': hbmass} 
        cells[df_name] = pd.DataFrame(dict)
        n=n+1
    
    return cells, cell_paths

def sat_data(cells, cell_paths):
    i=0
    cell_sats = []
    cell_vols =[]
    cell_mass=[]


    while i<len(cells):
        saturations=cells[list(cells)[i]]['sat']
        sat=np.stack( saturations, axis=0 )
        # vols=cells[list(cells)[i]]['vol']
        # vols=np.stack( vols, axis=0 )
        # vols = vols[~np.isnan(vols)]
        mass=cells[list(cells)[i]]['mass']
        mass=np.stack( mass, axis=0 )
        mass = mass[~np.isnan(mass)]
        sat[(sat > 1) | (sat < -1)] = np.nan
        d = {'sat': sat}#, 'preds': preds}
        sat_df = pd.DataFrame(data=d)
        sat_df['sat_mean']=sat_df['sat'].rolling(window=10).mean() 
        sat_mean=sat_df['sat_mean'].to_list()
        n = 355
        fill = [np.nan] * n
        sats = sat_mean
        cell_sats.append(sats)
        # cell_vols.append(vols)
        cell_mass.append(mass)
        i=i+1
    saturations_df= pd.DataFrame(cell_sats)
    saturations_df=saturations_df.T
    saturations_df.columns = cell_paths

    return saturations_df

def moving_norm(sat_df,norm0, norm21):
    xnorm=np.linspace(0, len(norm0),len(norm0))
    # plt.plot(xnorm, norm0, '--', label="fitted")
    # plt.plot(xnorm, norm21, '--', label="fitted")

    sat_level=sat_df.copy(deep=True)
    for column in sat_df:
        (sat_level[column])=((sat_level[column])-norm0[0:len((sat_level[column]))])/norm21[0:len((sat_level[column]))]

    sat_roll=sat_level.copy(deep=True)
    sat_roll=sat_roll.rolling(window=10).mean() 

    return sat_roll


def plot_data(sats, sig, number_of_subplots, number_of_columns):
    total = number_of_subplots
    cols = number_of_columns
    rows = total // cols
    if total % cols != 0:
        rows += 1

    fig, axs = plt.subplots(rows,cols, figsize=(cols*3, rows*3), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    i=0
    path=[]
    taus=[]
    sat_fin=[]
    sat_start=[]
    start=[]
    sat_drop=[]
    for series_name, series in sats.items():
        y=np.asarray(series[10:-50])
        ys=y[~np.isnan(y)]
        xs=np.linspace(0, (2*len(ys)/333),len(ys))

        if sig==True:
            try:
                p0 = [150, min(ys), max(ys),0.05]
                popt,pcov = curve_fit(piecewise_exponential, xs, ys,p0, method='dogbox')
                x0, P, Yo, k = popt
                sampleRate = 333 # Hz
                tauSec = (1 / k) / sampleRate
                tauS='{:.3}'.format(tauSec)
                path.append(series_name)
                taus.append(tauSec)
                sat_fin.append(P)
                sat_start.append(Yo)
                start.append(x0)
                sat_drop.append(Yo-P)


                # determine quality of the fit
                squaredDiffs = np.square(ys - piecewise_exponential(xs,x0, P, Yo, k ))
                squaredDiffsFromMean = np.square(ys - np.mean(ys))
                rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
                if rSquared>0.1:
                    rsq='{:.3}'.format(rSquared)
                    axs[i].plot(xs, ys, '.', label="data")
                    axs[i].plot(xs, piecewise_exponential(xs,x0, P, Yo, k ), '--', label="fitted")
                    axs[i].set_ylim([0, 1])
                    axs[i].text(150,0.7,f"R² = {rsq}" , fontsize=10)
                    axs[i].text(150,0.6,f"tau = {tauS}" , fontsize=10)
                else:
                    pass
            except RuntimeError:
                print("Error - curve_fit failed")
                axs[i].plot(xs, ys, '.', label="data")
                axs[i].set_ylim([0, 1])
        else:
            try:
                sat_fin.append(np.nanmean(y))
                p1 = [-0.01,00]
                popt,pcov = curve_fit(linear, xs, ys,p1, method='dogbox')
                m, b = popt
                slp='{:.3e}'.format(m)
                squaredDiffs = np.square(ys - linear(xs,m,b))
                squaredDiffsFromMean = np.square(ys - np.mean(ys))
                rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
                rsq='{:.3}'.format(rSquared)
                axs[i].plot(xs, ys, '.', label="data")
                axs[i].plot(xs, linear(xs,m,b), '--', label="fitted")
                axs[i].set_ylim([-0.1, 1.1])
                axs[i].text(3,0.7,f"R² = {rsq}" , fontsize=10)
                axs[i].text(3,0.6,f"slope = {slp}" , fontsize=10)
            except RuntimeError:
                print("Error - curve_fit failed")
                pass
        i=i+1
    d = {'name': path, 'tau': taus,'start': start, 'sat_start': sat_start, 'sat_fin': sat_fin, 'sat_drop': sat_drop}
    # tau_df = pd.DataFrame(data=d)