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
from scipy.ndimage import interpolation
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from PIL import Image
import random
import shutil
import pathlib
import keras
import os
import ruptures as rpt
from keras import layers
import seaborn as sns
import tensorflow as tf
from PIL import Image
import ipyplot
import glob
from PIL import Image
from natsort import natsorted
from scipy.ndimage.morphology import binary_dilation


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

def rescale(img):
    im=np.zeros([81,81])
    try:
        minval = np.min(img[np.nonzero(img)])
        maxval = np.max(img[np.nonzero(img)])
        im_range = maxval-minval
        if im_range == 0:
            im_range = 1
        im = (img - minval)/im_range 

    except ValueError:  #raised if `y` is empty.
        pass

    return im

def padding(array, xx, yy):

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    if a>0 and b> 0 and aa>0 and bb>0:
        padded=np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')
    else:
        padded=(np.zeros([81,81]))
    return padded

def euler(pt,pt_old): #This function is used to check if a cell found in a frame is close to the cell in the previous frame, to avoid getting different cells
    # print(len(pt))
    x_old=pt_old[0]; y_old=pt_old[1]
    if len(pt)==2:
        dist1 = math.hypot(x_old - pt[0][0], y_old - pt[0][1])
        dist2 = math.hypot(x_old - pt[1][0], y_old - pt[1][1])
        dist=[dist1,dist2]
        if np.argmin(dist)==0:
            point=pt[0]
        elif np.argmin(dist)==1:
            point=pt[1] 
    elif len(pt)==3:
        dist1 = math.hypot(x_old - pt[0][0], y_old - pt[0][1])
        dist2 = math.hypot(x_old - pt[1][0], y_old - pt[1][1])
        dist3 = math.hypot(x_old - pt[2][0], y_old - pt[2][1])
        dist=[dist1,dist2, dist3]
        if np.argmin(dist)==0:
            point=pt[0]
        elif np.argmin(dist)==1:
            point=pt[1] 
        elif np.argmin(dist)==2:
            point=pt[2]
    elif len(pt)==4:
        dist1 = math.hypot(x_old - pt[0][0], y_old - pt[0][1])
        dist2 = math.hypot(x_old - pt[1][0], y_old - pt[1][1])
        dist3 = math.hypot(x_old - pt[2][0], y_old - pt[2][1])
        dist4 = math.hypot(x_old - pt[3][0], y_old - pt[3][1])
        dist=[dist1,dist2, dist3, dist4]
        if np.argmin(dist)==0:
            point=pt[0]
        elif np.argmin(dist)==1:
            point=pt[1] 
        elif np.argmin(dist)==2:
            point=pt[2]
        elif np.argmin(dist)==3:
            point=pt[3]  
    else:
        point=pt_old
    # print(pt)
    return point

def euler(pt,pt_old): #This function is used to check if a cell found in a frame is close to the cell in the previous frame, to avoid getting different cells
    x_old=pt_old[0]; y_old=pt_old[1]
    if len(pt)==2:
        dist1 = math.hypot(x_old - pt[0][0], y_old - pt[0][1])
        dist2 = math.hypot(x_old - pt[1][0], y_old - pt[1][1])
        dist=[dist1,dist2]
        if np.argmin(dist)==0:
            point=pt[0]
        elif np.argmin(dist)==1:
            point=pt[1] 
    elif len(pt)==3:
        dist1 = math.hypot(x_old - pt[0][0], y_old - pt[0][1])
        dist2 = math.hypot(x_old - pt[1][0], y_old - pt[1][1])
        dist3 = math.hypot(x_old - pt[2][0], y_old - pt[2][1])
        dist=[dist1,dist2, dist3]
        if np.argmin(dist)==0:
            point=pt[0]
        elif np.argmin(dist)==1:
            point=pt[1] 
        elif np.argmin(dist)==2:
            point=pt[2]
    elif len(pt)==4:
        dist1 = math.hypot(x_old - pt[0][0], y_old - pt[0][1])
        dist2 = math.hypot(x_old - pt[1][0], y_old - pt[1][1])
        dist3 = math.hypot(x_old - pt[2][0], y_old - pt[2][1])
        dist4 = math.hypot(x_old - pt[3][0], y_old - pt[3][1])
        dist=[dist1,dist2, dist3, dist4]
        if np.argmin(dist)==0:
            point=pt[0]
        elif np.argmin(dist)==1:
            point=pt[1] 
        elif np.argmin(dist)==2:
            point=pt[2]
        elif np.argmin(dist)==3:
            point=pt[3]  
    else:
        point=pt_old
    return point

def find_cell_x(frame, pt_old):
    pts=[]
    found=True
    img=frame.astype('uint8')
    blur = 255-cv.GaussianBlur(img,(15,15),0)
    threshold = np.max(blur)*0.95
    _, binary = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY) 
# Find contours
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Filter by area to remove small noise artifacts
        if cv.contourArea(contour) > 1:  # Adjust minimum area as needed
            x, y, w, h = cv.boundingRect(contour)
            pts.append([x,y])
            # cv.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if len(pts)>1:
        pt=euler(pts,pt_old)
    elif len(pts)==1:
        pt=pts[0]
    else:
        pt[pt_old]
        found=False
    ptx=pt[0]; pty=pt[1]
    return ptx, pty, found 

def mass(Hb):
    parea=(6.9/20)**2; 
    try:
        Gy = cv.Sobel(Hb, cv.CV_64F, 0, 1, ksize=3)
    except ValueError:  #raised if `y` is empty.
        E=np.nan
        Hbnorm-np.zeros_like(Hb)
        pass    
    aGy=abs(Gy)
    means=(np.mean(aGy, axis=0))
    fwhm=(np.max(means))/2
    x=np.delete(Hb,np.where(means>fwhm), axis=1)
    means=(np.mean(x, axis=1))

    Hbnorm=Hb/means[:,None]
    Hbnorm[Hbnorm <= 0] = 1
    Hbnorm[Hbnorm >= 1] = 1

    E=((parea*(1e-8)*64500*np.sum(np.sum((-np.log10(Hbnorm))))))
    return E, Hbnorm

   
#Calcuate oxy and deoxy hemoglobin mass and cell saturation
def saturation(frames): 
     #camera pixel area
    b=[frames[0],frames[1]]
    #Molecular absorbtion coefficients
    w430_o = 2.1486*(10**8)
    w430_d = 5.2448*(10**8)
    w410_o = 4.6723*(10**8)
    w410_d = 3.1558*(10**8)

    E410, B410=mass(b[1])
    E430, B430=mass(b[0])
    B=[B430, B410]
    e=E410 #410
    f=E430 #430

    a=w410_d
    b=w410_o
    c=w430_d
    d=w430_o

    Mo=(e*c-a*f)/(b*c-a*d)
    Md=(b*f-e*d)/(b*c-a*d)

    saturation = Mo/(Mo+Md)
    hbmass=4*(Mo+Md)

    return saturation, hbmass, B

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
    top = [np.mean(-np.partition(-(frame[2].flatten()), 10)[:10]),np.mean(-np.partition(-(frame[3].flatten()), 10)[:10])]
    xpt430, ypt430, found430=find_cell_x(frame[0],[x_old,y_old])
    xpt410, ypt410, found410=find_cell_x(frame[1],[x_old,y_old])
    frame=[frame[0][:,int(xpt430-20):int(xpt430+20)],frame[1][:,int(xpt410-20):int(xpt410+20)],frame[2][:,int(xpt430-20):int(xpt430+20)],frame[3][:,int(xpt410-20):int(xpt410+20)]]

    b430=(frame[0])
    r430=(frame[2])
    pt430=[ypt430,xpt430]
    y430=pt430[0]
    b410=(frame[1])
    r410=(frame[3])
    pt410=[ypt410,xpt410]
    y410=pt410[0]; 
    if y430<15:
        y430=15
    if y410<15:
        y410=15

    frames=[b430[int(y430)-15:int(y430)+15, :],b410[int(y410)-15:int(y410)+15, :],r430[int(y430)-15:int(y430)+15, :],r410[int(y410)-15:int(y410)+15, :]]
    return frames, top, pt430, pt410, found430, found410

def masking(cell):
    cell = (255*cell).astype(np.uint8)
    kernel = np.ones((9,9),np.uint8)
    base=np.min(cell)+np.max(cell)/5
    _,mask= cv.threshold(cell,base,255,cv.THRESH_BINARY)
    closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return closing

def center(cell, mask):
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    im_pad=np.zeros((81,81))
    mask_pad=np.zeros((81,81))
    out = np.zeros_like(mask)

    if len(contours)>0:
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area>300:
                x, y, w, h = cv.boundingRect(cnt)
                hull=cv.convexHull(cnt)
                im_masked=cell#
                mask=cv.drawContours(out, [hull], -1, (255, 255, 255), -1)
                
                im_center=im_masked[int(y):int(y)+h,int(x):int(x)+w]
                mask_center=mask[int(y):int(y)+h,int(x):int(x)+w]
                im_pad=padding(im_center, 81, 81)
                mask_pad=padding(mask_center, 81, 81)
    else:
        im_pad=np.zeros((81,81))
        mask_pad=np.zeros((81,81))

    return im_pad, mask_pad, mask


def volume(cellR, mask, top):
    vol=np.nan
    channel=8
    parea=(6.9/20)**2; 
    cell=(cellR*mask)/255
    back=(cellR*(255-mask))/255
    back[back > top-50] = 0
    cellR[back > top-50] = 0
    means = back[back!=0].mean()
    transmission = means/top
    cell=cell/means
    cell[cell <= 1] = 1
    alf = -np.log(transmission)/channel
    tcell=np.log(cell)/alf
    tcell[tcell <= 0] = 0
    vol=((np.sum(np.sum(tcell)))*parea)/2

    return vol

def net(B, R,top):

    B430 = B[0]
    B410 = B[1]
    R430 = R[0]
    R410 = R[1]
    
    B430=cv.resize(B430, dsize=(162, 121), interpolation=cv.INTER_LINEAR)
    B410=cv.resize(B410, dsize=(162, 121), interpolation=cv.INTER_LINEAR)
    R430=cv.resize(R430, dsize=(162, 121), interpolation=cv.INTER_LINEAR)
    R410=cv.resize(R410, dsize=(162, 121), interpolation=cv.INTER_LINEAR)

    imgB430=1-B430
    imgB410=1-B410
    mask430=masking(imgB430)
    mask410=masking(imgB410)

    imgR430=R430
    imgR410=R410
    

    imgB430p, masks430, mask430=center(imgB430, mask430)
    imgB410p, masks410, mask410=center(imgB410, mask410)
    imgR430p, _, _=center(imgR430, mask430)
    imgR410p, _, _=center(imgR410, mask410)

    vol430=volume(imgR430,mask430,top[0])
    vol410=volume(imgR430,mask430,top[1])
    vol=(vol430+vol410)/2

    masks430=(masks430/255).astype(np.float64)
    masks410=(masks410/255).astype(np.float64)

    imgB430p=(imgB430p*masks430)
    imgB410p=(imgB410p*masks410)
    imgR430p=(imgR430p*masks430)
    imgR410p=(imgR410p*masks410)

    imgB430p=rescale(imgB430p)
    imgB410p=rescale(imgB410p)
    imgR430p=rescale(imgR430p)
    imgR410p=rescale(imgR410p)

    imgB410p[imgB410p<0] = 0
    imgB430p[imgB430p<0] = 0
    imgR410p[imgR410p<0] = 0
    imgR430p[imgR430p<0] = 0
    masks410[masks410<0] = 0
    masks430[masks430<0] = 0

    imgR=imgR430p/2+imgR410p/2

    img = np.zeros([81,81,3])
    img[:,:,0] = imgB410p
    img[:,:,1] = imgB430p
    img[:,:,2] = imgR

    return img, vol, imgB410p, imgB430p, imgR410p, imgR430p, masks410, masks430

def main_run(video):
    BL=getBL()
    i=0
    x_old=0
    y_old=0
    saturations=[]
    imgs=[]
    MCV=[]
    hgb=[]
    x=[0]*10
    y=[0]*10
    I430=[]
    I410=[]
    while i<len(video):
        # print(i)
        frame1=video[i][0:2]
        frame2=video[i][2:4]
        frame3=video[i][2:6]
        frame4=video[i][6:8]
        frames=[frame1,frame2, frame3, frame4]
        for frame in frames:
            x_old=np.mean(x[-9:])
            y_old=np.mean(y[-9:])
            frame, top, pt430, pt410, found430, found410 = segment(frame,x_old, y_old,BL)
            if pt430[1]>=21 and pt410[1]>21 and pt430[1]<339 and pt410[0]<339 and pt430[0]>11 and pt410[0]>11 and pt430[0]<239 and pt410[0]<239 and found430==True and found410== True:
                sats, hbmass, B=saturation(frame)
                img, vol, imgB410, imgB430, imgR410, imgR430, mask410, mask430=net(B,frame[2:4], top)
                if img.size==19683 and hbmass < 5e-10 and hbmass > -0 and vol <300 and vol >0:
                    saturations.append(sats)
                    hgb.append(hbmass)
                    imgs.append(img)
                    MCV.append(vol)
                    x.append(pt430[1])
                    y.append(pt430[0])
                else:
                    saturations.append(np.nan)
                    MCV.append(np.nan)
                    hgb.append(np.nan)
                    imgs.append(np.zeros([81,81,3]))
                    x.append(300)
                    y.append(0)
            else:
                saturations.append(np.nan)
                MCV.append(np.nan)
                hgb.append(np.nan)
                imgs.append(np.zeros([81,81,3]))
                x.append(300)
                y.append(0)
                # I430.append(np.nan)
                # I410.append(np.nan)
        i=i+1
    return imgs, saturations, MCV, x, hgb

def view(video):
    # BL=getBL()
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
            L1=frame[0][1::2, 1::2]
            L2=frame[1][1::2, 1::2]
            if np.mean(L1)>np.mean(L2):
                b=L1
            else:
                b=L2
            x_old=np.mean(x[-3:])
            y_old=np.mean(y[-3:])
            # frame, _,_,_,_,_ = segment(frame,x_old, y_old,BL)
            gray=cv.resize(b, dsize=(720, 540), interpolation=cv.INTER_CUBIC).astype("uint8")
            cv.imshow('Single Track', gray)
            cv.waitKey(10)
        i=i+1
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
        frame1=video[i][0:2]
        frame2=video[i][2:4]
        frame3=video[i][2:6]
        frame4=video[i][6:8]
        frames=[frame1,frame2, frame3, frame4]
        for frame in frames:
            gray=cv.resize(frame[0][1::2, 1::2], dsize=(720, 540), interpolation=cv.INTER_CUBIC).astype("uint8")
        # gray = cv.cvtColor(frame[0], cv.COLOR_BayerBG2GRAY)
        # if pt430[0]>21 and pt410[0]>21 and pt430[0]<349 and pt410[0]<349 and pt430[1]>11 and pt410[1]>11 and pt430[1]<239 and pt410[1]<239:
            # frames, top=crop(frame, pt430, pt410)
            # sat, hb=saturation(frames)
            # pt430= np.asarray(pt430, dtype=np.float32)
            # pt410= np.asarray(pt410, dtype=np.float32)
            # ptres430=2*pt430; ptres410=2*pt410
            # img = gray[y-40:y+40,x-40:x+40]
            # cv.rectangle(gray, (locx-20, locy-20), (locx+20, locy+20), (255,100,200),2)
            # # img=cv.resize(img, dsize=(200, 200), interpolation=cv.INTER_CUBIC).astype("uint8")
            # cv.putText(gray, str('%f' %i), (45,45), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            vid.write(gray)
        i=i+1
    # print(xf)
    cv.destroyAllWindows()
    vid.release()

# #Chunk the video into N frame chunks for the CNN
# #Chunk the video into N frame chunks for the CNN
# def chunk(video, num):
#     i=0
#     stack=[]
#     mod=len(video)%num
#     length=(len(video)-mod)/num
#     chunked=np.array_split(video, length)
#     while i<length:
#         j=0
#         clip=[]
#         frames = chunked[i]
#         while j<num:
#             clip.append((255*frames[j]).astype(np.uint8))
#             j=j+1    
#         stack.append(clip)    
#         i=i+1
#     return stack

# #Save the chunk for CNN anaylsis
# def save_stack(stack, name, num):
#     # print(len(stack))
#     i=0
#     imgs=[]
#     # vid = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'), 10, (81, 81), True)
#     # while i<19:
#     while i<len(stack):
#         img = cv.normalize(stack[i], None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype("uint8")
        
#         if np.max(img[:,:,0])>0 and np.max(img[:,:,1])>0 and np.max(img[:,:,2])>0 and np.min(img[:,:,0])==0 and np.min(img[:,:,1])==0 and np.min(img[:,:,2])==0:
#         # img=cv.resize(norm, dsize=(100, 100), interpolation=cv.INTER_CUBIC).astype("uint8")
#             imgs.append(img)
#         i=i+1
#     # print(len(imgs))
#     if len(imgs)==len(stack):
#         j=0
#         vid = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'), num, (81, 81), True)
#         while j<len(imgs):
#             vid.write(imgs[j])
#             j=j+1
#         vid.release() 

# def partition(alist, indices):
#     return [alist[i:j] for i, j in zip([0]+indices, indices+[None])]

def CNN(imgs, model):
    np_imgs=np.array(imgs)
    pred=[]
    i=0
    batchsize=32
    for i in range(0, len(np_imgs), batchsize):
        X_batch = np_imgs[i: i+batchsize]
        X_batch=tf.convert_to_tensor(X_batch, dtype=tf.float32)
        try:
            predictions = model.predict(X_batch)
            pred.append(predictions)
        except:
            pred.append(np.nan)
            pass
        i=i+1
    predict=np.concatenate(pred)
    return predict

###ANAYLSIS###

def batch_analyze(file_path):
    cells = {}
    cell_paths=[]
    model=keras.models.load_model('CNN/model_ResNet50_A01.h5', safe_mode=False) #load the model
    files = glob.glob(file_path, 
                    recursive = True)
    files=natsorted(files)
    n=0
    for file in files:
        print(file)
        double=False
        with h5py.File(file, 'r') as hf:
            video = hf['data'][:]
        imgs, saturations, MCV, x, MCH = main_run(video)
        x_df= pd.DataFrame(x)
        x_df=x_df.dropna()
        x_list=x_df[0].to_list()
        # double=double_step(x_list)
        if double==False:
            saturations=np.asarray(saturations)
            preds=CNN(imgs,model)
            ps = [vec[0] for vec in preds]
            MCV=np.asarray(MCV)
            df_name='df0_'+str(n)
            cell_paths.append(file)
            dict = {'sat': saturations, 'MCV': MCV, 'MCH': MCH, 'pred': ps} 
            cells[df_name] = pd.DataFrame(dict)
        n=n+1
    
    return cells, cell_paths

def batch_image(file_path, dest_path):
    cells = {}
    cell_paths=[]
    files = glob.glob(file_path, 
                    recursive = True)
    files=natsorted(files)
    n=0
    for file in files:
        print(file)
        name=(os.path.basename(file)[0:-3])
        print(name)
        double=False
        with h5py.File(file, 'r') as hf:
            video = hf['data'][:]
        imgs, saturations, MCV, x, MCH = main_run(video)
        x_df= pd.DataFrame(x)
        x_df=x_df.dropna()
        x_list=x_df[0].to_list()
        # double=double_step(x_list)
        if double==False:
            cells=[]
            i=0
            # while i<len(imgs):
            for img in imgs:
                if img.size==19683 and MCH[i] < 5e-10 and MCH[i] > -0 and MCV[i]<200 and MCV[i] >20:
                    cells.append(img)
                i=i+1
            j=0
            while j<len(cells):
                A = cells[j]
                filename=dest_path+name+'_%d.png'%j
                tf.keras.utils.save_img(filename, A)
                j=j+1
        n=n+1

# def batch_video(file_path):
#     cells = {}
#     cell_paths=[]
#     files = glob.glob(file_path, 
#                     recursive = True)
#     files=natsorted(files)
#     n=0
#     for file in files:
#         print(file)
#         name=(os.path.basename(file)[0:-3])
#         print(name)
#         double=False
#         with h5py.File(file, 'r') as hf:
#             video = hf['data'][:]
#         imgs, saturations, MCV, x, MCH = main_run(video)
#         x_df= pd.DataFrame(x)
#         x_df=x_df.dropna()
#         x_list=x_df[0].to_list()
#         # double=double_step(x_list)
#         if double==False:
#             saturations=np.asarray(saturations)
#             num=20
#             stack=chunk(imgs,num)
#             i=0
#             while i<len(stack):
#                 if len(stack[i])==num:
#                     padded_num = str(i).rjust(3, '0')
#                     fname='CNN/tmp/data/'+name+padded_num+'.avi'
#                     save_stack(stack[i],fname, num)
#                 # else:
#                     #todo add place holder
#                 i=i+1
#         n=n+1

def subsample_images(source_folder, destination_folder, sub_fraction):
    """
    Subsamples a specified number of images from a source folder and copies them
    to a destination folder.

    Args:
        source_folder (str): The path to the folder containing the original images.
        destination_folder (str): The path to the folder where subsampled images
                                  will be copied.
        num_samples (int): The number of images to subsample.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    image_files = [f for f in os.listdir(source_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    num_samples=int(len(image_files)*sub_fraction)
    if num_samples > len(image_files):
        print(f"Warning: Requested {num_samples} samples, but only {len(image_files)} images available.")
        num_samples = len(image_files)

    selected_files = random.sample(image_files, num_samples)

    for filename in selected_files:
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copy(source_path, destination_path)

def find_step(ps):
    y=np.asarray(ps)
    dary=y[~np.isnan(y)]
    dary -= np.average(dary)
    step = np.hstack((np.ones(len(dary)), -1*np.ones(len(dary))))
    dary_step = np.convolve(dary, step, mode='valid')
    step_indx = np.argmax(abs(dary_step))
    if dary_step[step_indx]>150:
        pos=step_indx +100
    else:
        pos=np.nan
    return pos

def sat_data(cells, cell_paths,threshold):
    i=0
    cell_sats = []
    cell_vols =[]
    cell_mass=[]
    cell_conc=[]
    cell_preds=[]
    while i<len(cells):
        pred=cells[list(cells)[i]]['pred']
        preds=np.stack( pred, axis=0 )
        saturations=cells[list(cells)[i]]['sat']
        sat=np.stack( saturations, axis=0 )
        MCVs=cells[list(cells)[i]]['MCV']
        MCVs=np.stack( MCVs, axis=0 )
        MCHs=cells[list(cells)[i]]['MCH']
        MCHs=np.stack( MCHs, axis=0 )
        MCHC=(MCHs/MCVs)*(10**14)
        sat[(sat > 1) | (sat < -1)] = np.nan
        MCVs[(MCVs > 200) | (MCVs < 0)] = np.nan
        MCHs[(MCHs > 5e-10) | (MCHs < -0)] = np.nan
        MCHs=MCHs*1e12
        d = {'sat': sat, 'MCV': MCVs, 'MCH': MCHs, 'MCHC': MCHC, 'preds': preds}
        df = pd.DataFrame(data=d)
        # df=df.dropna()
        cell_sats.append(df['sat'].to_numpy())
        cell_vols.append(df['MCV'].to_numpy())
        cell_mass.append(df['MCH'].to_numpy())
        cell_conc.append(df['MCHC'].to_numpy())
        cell_preds.append(df['preds'].to_numpy())
        i=i+1
    saturations_df= pd.DataFrame(cell_sats)
    # saturations_df.reset_index(drop=True, inplace=True)
    saturations_df=saturations_df.T
    saturations_df.columns = cell_paths

    vols_df= pd.DataFrame(cell_vols)
    vols_df=vols_df.T
    vols_df.columns = cell_paths

    mass_df= pd.DataFrame(cell_mass)
    mass_df=mass_df.T
    mass_df.columns = cell_paths

    conc_df= pd.DataFrame(cell_conc)
    conc_df=conc_df.T
    conc_df.columns = cell_paths
    
    pred_df= pd.DataFrame(cell_preds)
    pred_df=pred_df.T
    pred_df.columns = cell_paths

    classified=[]
    pos=[]
    for series_name, series in pred_df.items():
        ps=np.array(series)
        sickle=find_step(ps)
        if sickle < 1100 and sickle > 500:
            sickled=True
        else:
            sickled=False
        classified.append(sickled)
        if sickle > 500:
            pos.append(sickle)
        else:
            pos.append(np.nan)
    class_df= pd.DataFrame(classified)
    class_df=class_df.T
    class_df.columns = cell_paths

    pos_df= pd.DataFrame(pos)
    pos_df=pos_df.T
    pos_df.columns = cell_paths

    return saturations_df, vols_df, mass_df, conc_df, pred_df, class_df, pos_df

def scale_data(df, size):
    r = pd.RangeIndex(0, size+3, 1)
    df = df.sort_index()
    new_idx = np.linspace(df.index[0], df.index[-1], len(r))
    df= (df.reindex(new_idx, method='ffill', limit=1).iloc[1:].interpolate())
    df=df.reset_index()
    df=df.drop('index', axis=1)
    df=df.squeeze()
    return df

def moving_norm(sat_df,norm0, norm21):
    norm0=scale_data(norm0,len(sat_df))
    norm21=scale_data(norm21,len(sat_df))
    sat_level=sat_df.copy(deep=True)
    for column in sat_df:
        (sat_level[column])=((sat_level[column])-norm0[0:len((sat_level[column]))])/(norm21[0:len((sat_level[column]))]-norm0[0:len((sat_level[column]))])
    sat_roll=sat_level.copy(deep=True)
    return sat_roll

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def filter(sat_df):
    x=sat_df.iloc[0:2000].values
    peaks, properties = find_peaks(x, prominence=(None, 0.2))
    filt=np.take(x, peaks, 0)
    z=len(x)/len(filt)
    filtd=interpolation.zoom(filt,z)
    return filtd    

def calibrate(sat_df):
    sat_cal=pd.DataFrame()
    for column in sat_df:
        sat_cal[column]=(sat_df[column])
        (sat_cal[column])=((sat_cal[column])-np.nanmean(sat_cal[column][-400:-100]))/(np.nanmean(sat_cal[column][100:400])-np.nanmean(sat_cal[column][-400:-100]))   
    return sat_cal

def plot_sat_data(sats, sig, number_of_subplots, number_of_columns):
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
        y=np.asarray(series[100:-700])
        ys=y[~np.isnan(y)]
        xs=np.linspace(0, (2*len(ys)/333),len(ys))

        if sig==True:
            try:
                p0 = [2.5, min(ys), max(ys),2]
                popt,pcov = curve_fit(piecewise_exponential, xs, ys,p0, method='dogbox')
                x0, P, Yo, k = popt
                sampleRate = 167 # Hz
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
                if rSquared>0.7:
                    rsq='{:.3}'.format(rSquared)
                    axs[i].plot(xs, ys, '.', label="data")
                    axs[i].plot(xs, piecewise_exponential(xs,x0, P, Yo, k ), '--', label="fitted")
                    axs[i].set_ylim([-0.1, 1.1])
                    axs[i].text(1,0.7,f"R² = {rsq}" , fontsize=10)
                    axs[i].text(1,0.6,f"tau = {tauS}" , fontsize=10)
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
                axs[i].plot(xs, ys, '-', label="data")
                # axs[i].plot(xs, linear(xs,m,b), '--', label="fitted")
                axs[i].set_ylim([-0.2, 1.1])
                # axs[i].text(3,0.7,f"R² = {rsq}" , fontsize=10)
                # axs[i].text(3,0.6,f"slope = {slp}" , fontsize=10)
            except RuntimeError:
                print("Error - curve_fit failed")
                pass
        i=i+1
    d = {'name': path, 'tau': taus,'start': start, 'sat_start': sat_start, 'sat_fin': sat_fin, 'sat_drop': sat_drop}
    # tau_df = pd.DataFrame(data=d)
    return d

def save(name, imgs):
    #track_vids/20250313_MGH2118/4/vids/20250313_MGH2118_4_3.h5
    with h5py.File(name,'w') as h5f:
        h5f.create_dataset("data", data=imgs)

def load(name):
    with h5py.File(name, 'r') as hf:
        video = hf['data'][:]
    return video