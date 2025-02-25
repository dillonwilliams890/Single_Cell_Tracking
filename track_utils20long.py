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
import tqdm
import einops
import random
import pathlib
import itertools
import collections
import keras
import os
import ruptures as rpt
from keras import layers
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
import pathlib
from tensorflow.keras.optimizers import Adam
import ipyplot
import glob
from PIL import Image
from natsort import natsorted
from CNN_utils import *


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

def findwall(pt,chan):
    if pt > chan[0] + 10 and pt < chan[1] - 10:
        y=pt
    elif pt > chan[0] + 10 and pt > chan[1] - 10:
        y=chan[1] - 12
    elif pt < chan[0] + 10 and pt < chan[1] - 10:
        y=chan[0] + 12
    else:
        y=pt
    return y

def crop(frame, pt430, pt410): 
    frames=[]
    c=[]
    top=[np.mean(frame[2][int(0):int(10), int(0):int(40)]),np.mean(frame[3][int(0):int(10), int(0):int(40)])]
    blue_top=top=[np.mean(frame[0][int(0):int(10), int(0):int(40)]),np.mean(frame[1][int(0):int(10), int(0):int(40)])]
    xs=[pt430[0],pt410[0],pt430[0],pt410[0]]
    # ys=[pt430[1],pt410[1],pt430[1],pt410[1]]
    i=0
    for f in frame[2:]:
        img=(f[int(0):int(270), int(xs[i]-20):int(xs[i]+20)])
        Gy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
        aGy=abs(Gy)
        imGy=np.copy(img)
        grad=np.median(aGy,axis=1)
        grad=np.where(grad > 30, 0, 1)
        wall=np.where(grad == 0)[0]
        gap=np.diff(wall)
        chan=[wall[np.argmax(gap)-1]+1, wall[np.argmax(gap)+1]-1]
        c.append(chan)
        i=i+1
    y430=findwall(pt430[1],c[0])
    y410=findwall(pt410[1],c[1])
    frames=[frame[0][int(y430)-10:int(y430)+10, int(xs[0]-20):int(xs[0]+20)],frame[1][int(y410)-10:int(y410)+10, int(xs[1]-20):int(xs[1]+20)],frame[2][int(y430)-10:int(y430)+10, int(xs[0]-20):int(xs[0]+20)],frame[3][int(y410)-10:int(y410)+10, int(xs[1]-20):int(xs[1]+20)]]
    # plt.imshow(frames[0])
    return frames, top, blue_top


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

def euler(f, thresh, x_old, y_old): #This function is used to check if a cell found in a frame is close to the cell in the previous frame, to avoid getting different cells
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

def mass(Hb):
    # x=pt[0]; y=pt[1]
    # parea=6.9
    parea=(6.9/20)**2; 
    # Gx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    base=np.min(Hb)+(np.max(Hb)-np.min(Hb))/3
    level, cell_mask = cv.threshold(Hb, base, 255, cv.THRESH_BINARY)
    # _, wall_mask = cv.threshold(Hb, 1.8*level, 255, cv.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv.erode(cell_mask,kernel,iterations = 1)
    kernel_outer = np.ones((5,5),np.uint8)
    dilation_outer = cv.erode(cell_mask,kernel_outer,iterations = 1)
    cell_mask=cell_mask/255
    dilation=dilation/255
    masked = (Hb*dilation)
    # plt.imshow(masked)
    Hb=Hb*(1-dilation_outer)
    
    average = masked[np.nonzero(masked)].mean()
    Hbnorm=Hb/average
    Hbnorm[Hbnorm <= 0] = 1
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
   
def saturation(frames): #Calcuate cell saturation
     #camera pixel area
    b=[frames[0],frames[1]]
    #Molecular absorbtion coefficints of something like that ~chemistry~
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
    Mo=(a*f-e*c)/(a*d-b*c)
    Md=(e*d-b*f)/(a*d-b*c)

    saturation = Mo/(Mo+Md)
    hbmass=e+f
    # print(Mo)
    # print('$')
    # print(Md)
    return saturation, hbmass

#resize the images, and process them for the neural net
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

# Seperate the frames and find the cell in each frame
def segment(frames,x_old, y_old,BL):
    thresh410=25
    thresh430=25
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
    f430 = tp.locate(b[0], 31, preprocess=True, percentile=99,  invert=True, max_iterations=1, characterize =True, topn=2)
    f410 = tp.locate(b[1], 31, preprocess=True, percentile=99,  invert=True, max_iterations=1, characterize =True, topn=2)
    # print(f430.head())
    pt430=euler(f430,thresh430, x_old, y_old)
    pt410=euler(f410,thresh410, x_old, y_old)
    frame=bleed(frame,BL)

    return pt430,pt410, frame

def getBL(): #Load the BL file
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

def main_run(video): #Run the anaylsis in the video
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
            pt430, pt410, frame = segment(frame,x_old, y_old,BL)
            # print(pt430)
            # print(pt410)
            if pt430[0]>21 and pt410[0]>21 and pt430[0]<339 and pt410[0]<339 and pt430[1]>11 and pt410[1]>11 and pt430[1]<239 and pt410[1]<239:
                frames, top, blue_top=crop(frame,pt430,pt410)
                sats, hbmass=saturation(frames)
                saturations.append(sats)
                hgb.append(hbmass)
                img, vol=net(frames, top)
                imgs.append(img)
                volumes.append(vol)
                x.append(pt410[0])
                y.append(pt410[1])
                I430.append(blue_top[0])
                I410.append(blue_top[1])
            else:
                saturations.append(np.nan)
                volumes.append(np.nan)
                hgb.append(np.nan)
                imgs.append(np.zeros([81,81,3]))
                x.append(0)
                I430.append(np.nan)
                I410.append(np.nan)
        i=i+1
    return imgs, saturations, volumes, x, hgb, I430, I410 

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
            pt430, pt410, frame = segment(frame,x_old, y_old,BL)
            gray=cv.resize(frame[1], dsize=(720, 540), interpolation=cv.INTER_CUBIC).astype("uint8")
            # print(pt430)
            # gray = cv.cvtColor(frame[0], cv.COLOR_BayerBG2GRAY)
            if pt430[0]>21 and pt410[0]>21 and pt430[0]<339 and pt410[0]<339 and pt430[1]>11 and pt410[1]>11 and pt430[1]<239 and pt410[1]<239:
                frames, top, b=crop(frame, pt430, pt410)
                sat, hb=saturation(frames)
                pt430= np.asarray(pt430, dtype=np.float32)
                pt410= np.asarray(pt410, dtype=np.float32)
                ptres430=2*pt430; ptres410=2*pt410
                x.append(pt410[0]); y.append(pt410[1])
                locx=int(ptres410[0]); locy=int(ptres410[1])
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

# def CNN(imgs): #Run the CNN and display the resutls
#     model=keras.models.load_model('CNN/model_ResNet50_A01.h5', safe_mode=False) #load the model
#     cnn_imgs=np.stack( imgs, axis=0 )
#     predictions = model.predict(cnn_imgs) #this is the line that calls the CNN to classify
#     # print(predictions[0:10])

#     threshold=0.5 #threshold is set at 0.5 to start
#     labels=[]
#     preds=[]
#     cell_imgs=[]
#     for i in range(len(predictions)):
#         preds.append(predictions[i][0])
#         cell_imgs.append(cnn_imgs[i][:,:,1])
#         if predictions[i][0] > threshold:
#             labels.append(0)
#         else:
#             labels.append(1)
#     preds=(predictions[:,0]) #get the first index of the predicitons, I believe this is what we've been using
#     preds=np.round(preds, 3)
#     pred=np.array(preds) 
#     sorted = pred.argsort() #sort the predicitons by descending order
#     sorted_preds = pred[sorted[::-1]]
#     sorted_cells = cnn_imgs[sorted[::-1]]
#     # ipyplot.plot_images(sorted_cells, sorted_preds,max_images=500, img_width=50)
#     return preds

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