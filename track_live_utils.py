
import numpy as np  
from pycromanager import Core
import time
from simple_pyspin import Camera
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2 as cv
import numpy as np
# import pandas as pd
import h5py
import matplotlib.pyplot as plt
import sys
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import time
import pims
import trackpy as tp
import numba
import scipy
import imageio
# from __future__ import print_function 
import time
import math
import flir
import seaborn as sns
from simple_pid import PID
from Fluigent.SDK import fgt_init, fgt_close
from Fluigent.SDK import fgt_set_pressure, fgt_get_pressure, fgt_get_pressureRange
import tkinter as tk


def initial(img, thresh):
    # frame=img
    signal=0
    
    cell=False
    L1=img[0][1::2, 1::2]
    L2=img[1][1::2, 1::2]
    if np.mean(L1)>np.mean(L2):
        b=[L1,L2]
        # frame=[img[0],img[1]]
    else:
        b=[L2,L1]
        # frame=[img[1],img[0]]
    L2=img[1][1::2, 1::2]
    if np.mean(L1)>np.mean(L2):
        b=[L1,L2]
    else:
        b=[L2,L1]
    f = tp.locate(b[0], 41, preprocess=True, percentile=99,  invert=True, max_iterations=1, characterize =True, topn=1)
    if len(f)<1:
        signal=0
    elif len(f)>0:
        f=f.max()
        signal=f.signal
        ecen=f.ecc
        x=f.x
    if signal>thresh  and x<150 and ecen<0.4:
        cell=True
    return cell

def feedback(frames,x_old, y_old, thresh):
    ecc=0.4
    pt=[0,0,0,0]
    ts=0
    img=frames
    L1=img[0][1::2, 1::2]
    L2=img[1][1::2, 1::2]
    if np.mean(L1)>np.mean(L2):
        b=[L1,L2]
    else:
        b=[L2,L1]
    f = tp.locate(b[0], 31, preprocess=True, percentile=99,  invert=True, max_iterations=1, characterize =True, topn=2)
    # gray = cv.cvtColor(b[0], cv.COLOR_BayerBG2GRAY)
    if len(f)<1:
        print('a')
        pt=[0,0,0,0]
        ts=0
    elif len(f)==1:
        pt=[f.x.values[0],f.y.values[0], f.signal.values[0],f.ecc.values[0]]
        ts = time.time()
    elif len(f)>1:
        pt1=[f.x.values[0],f.y.values[0], f.signal.values[0],f.ecc.values[0]]
        ts = time.time()
        pt2=[f.x.values[1],f.y.values[1], f.signal.values[1],f.ecc.values[1]]
        # print(pt1)
        # print(pt2)
        if pt1[2]>thresh and pt2[2]>thresh and pt1[3]<ecc and pt2[3]<ecc:
            dist1 = math.hypot(x_old - pt1[0], y_old - pt1[1])
            dist2 = math.hypot(x_old - pt2[0], y_old - pt2[1])
            if dist1<dist2:
                pt=pt1
            elif dist1>dist2:
                pt=pt2
        elif pt1[2]>thresh and pt2[2]<thresh and pt1[3]<ecc:
            pt=pt1
        elif pt1[2]<thresh and pt2[2]>thresh and pt2[3]<ecc:
            pt=pt2
        elif pt1[2]>thresh and pt2[2]>thresh and pt2[3]<ecc:
            pt=pt2
    else:
        # print('b')
        pt=[0,0,0,0]
        ts=0
    # print(pt)
    if pt[2]>thresh and pt[3]<ecc:
        pt=pt
        ts = time.time()
    else:
            # print('c')
            pt=[0,0,0,0]
            ts=0

    return pt, ts
def init(core):
    x_init=core.get_x_position()
    y_init=core.get_y_position()
    z_init=core.get_position()
    return x_init, y_init, z_init

def fin(core):
    x_fin=core.get_x_position()
    y_fin=core.get_y_position()
    z_fin=core.get_position()
    return x_fin, y_fin, z_fin


def set_to_start(core,pressure, x_init,y_init):
    core.set_xy_position(x_init,y_init)
    core.wait_for_device(core.get_xy_stage_device())
    fgt_init()
    fgt_set_pressure(0, pressure)


def run(core,pressure, x_init,y_init, x_fin,y_fin, thresh):
    press=pressure
    fgt_init()   
    fgt_set_pressure(0, press)
    core.set_xy_position(x_init,y_init)
    core.wait_for_device(core.get_xy_stage_device())
    i=0
    p=press
    imgs=[]
    locx=[]
    pressures=[]
    run=False
    cell=False
    x_old=0
    y_old=0
    x=[180,180,180]
    y=[145,145,145] 
    t=[]
    pid = PID(0.5, 0.01, 1, setpoint=200)
    pid.output_limits = (-2, 2)
    with Camera() as cam: # Acquire and initialize Camera
        # Start recording
        fgt_init()
        fgt_set_pressure(0, p)
        # core.wait_for_device(core.get_xy_stage_device())
        # while cell==False:
            # print(time.time())
        while run==False:
            cam.start()
            frame = [cam.get_array() for n in range(2)]
            cell=initial(frame, thresh)
            # print(cell)
            # print(time.time())
            if cell ==True:
                # cam.stop()
                core.set_xy_position(x_fin,y_fin) 
                # time.sleep(0.01)
                cam.start()  
                while  cell==True and core.get_x_position()<x_fin-1:
                    x_old=np.mean(x[-3:])
                    # print(x_old)
                    y_old=np.mean(y[-3:])
                    img = [cam.get_array() for n in range(8)] # Get 10 frames
                    # frame = img#np.maximum(data2[i][0],data2[i][1])
                    pt, ts = feedback(img,x_old, y_old, thresh)
                    x_pt=pt[0]; y_pt=pt[1]
                    imgs.append(img)
                    drift=2*0.1725*(x-x_old)
                    x_pid=np.mean(x[-3:])
                    pressure=pid(x_pid)
                    locx.append(x_pt)
                    x.append(x_pt)
                    y.append(y_pt)
                    print(x_pt)
                    t.append(i)
            # print(drift)
                    if  ts > 0 and p>10 and  p<1499:
                        
                    #     p=p+2
                    #     fgt_set_pressure(0, p)
                    # elif drift>2 and drift<50 and ts > 0 and  p>100 and  p<1499:
                    #     p=p-3
                    #     fgt_set_pressure(0, p)
                        if p<press-100:
                            p=press-100
                    #     fgt_set_pressure(0, p)
                        elif p>press+100:
                            p=press+100
                        else:
                            p+=pressure
                    
                    else:
                        p=p

                # print(time.time())
                    pressures.append(p)
                    fgt_set_pressure(0, p)
                    # x_old=x
                    # y_old=y
                    i=i+1
                    # cv.rectangle(img[0], (pt[1]-20, pt[1]-20), (pt[0]+20, pt[0]+20), (255,100,200),2)
                    # cv.imshow('Single Track', img)
        # # press 'q' to bre
                

                run=True
            else:
                run==False
            cam.stop()
    cv.destroyAllWindows()
    plt.plot(locx)   
    return imgs, pressures    

def save(name, imgs):
    #track_vids/20250313_MGH2118/4/vids/20250313_MGH2118_4_3.h5
    with h5py.File(name,'w') as h5f:
        h5f.create_dataset("data", data=imgs)

def load(name):
    with h5py.File(name, 'r') as hf:
        video = hf['data'][:]
    return video

def veiw_50(imgs):
    x=[0,0,0]
    y=[0,0,0] 
    i=0         
    while i<50:
        img=imgs[i]
        cell=initial(img)
        print(cell)
        # gray = cv.cvtColor(frame, cv.COLOR_BayerBG2GRAY)
        L1=img[0][1::2, 1::2]
        L2=img[1][1::2, 1::2]
        print(np.mean(L1))
        # b1=np.max([L1, L2],axis=0)
        # b2=np.min([L1, L2],axis=0)
        if np.mean(L1)>np.mean(L2):
            b=[L1,L2]
        else:
            b=[L2,L1]
        
        f = tp.locate(b[0], 31, preprocess=True, percentile=99,  invert=True, max_iterations=1, characterize =True, topn=1)
        # t=f[f['signal']==f['signal'].max()]
        
        frame = cv.cvtColor(b[0], cv.COLOR_BayerBG2GRAY)
        # # print(signal)
        signal=f.signal.values[0]
        ecc=f.ecc.values[0]
        xf=f.x.values[0]
        yf=f.y.values[0]
        # signal=int(pt[2])
        x_pt=int(xf)
        y_pt=int(yf)
        # if signal>14:
        cv.rectangle(frame, (x_pt-20, y_pt-20), (x_pt+20, y_pt+20), (255,100,200),2)
        cv.putText(frame, str('%f' %ecc), (x_pt, y_pt), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv.putText(frame, str('%f' %signal), (x_pt, y_pt - 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            # drift=0.1725*(x-x_old)
        cv.imshow('Single Track', frame)
        # # press 'q' to bre
        cv.waitKey(0)
        i=i+1
    cv.destroyAllWindows()
     # %%
# %%

#%%
def collect(core, name, pressure, start, end,x_init,y_init, x_fin, y_fin, thresh):
    n=start
    while n<end:
        press=pressure
        fgt_init()   
        fgt_set_pressure(0, press)
        core.set_xy_position(x_init,y_init)
        core.wait_for_device(core.get_xy_stage_device())
        i=0
        p=press
        imgs=[]
        locx=[]
        pressures=[]
        run=False
        cell=False
        x_old=0
        y_old=0
        x=[0,0,0]
        y=[0,0,0] 
        t=[]
        pid = PID(0.5, 0.01, 1, setpoint=200)
        pid.output_limits = (-2, 2)
        with Camera() as cam: # Acquire and initialize Camera
            # Start recording
            fgt_init()
            fgt_set_pressure(0, p)
            # core.wait_for_device(core.get_xy_stage_device())
            # while cell==False:
                # print(time.time())
            while run==False:
                cam.start()
                frame = [cam.get_array() for n in range(2)]
                cell=initial(frame, thresh)
                # print(time.time())
                if cell ==True:
                    # cam.stop()
                    core.set_xy_position(x_fin,y_fin) 
                    # time.sleep(0.01)
                    cam.start()  
                    while  cell==True and core.get_x_position()<x_fin-1:
                        x_old=np.mean(x[-3:])
                        # print(x_old)
                        y_old=np.mean(y[-3:])
                        img = [cam.get_array() for n in range(8)] # Get 10 frames
                        # frame = img#np.maximum(data2[i][0],data2[i][1])
                        pt, ts = feedback(img,x_old, y_old, thresh)
                        x_pt=pt[0]; y_pt=pt[1]
                        imgs.append(img)
                        drift=2*0.1725*(x-x_old)
                        x_pid=np.mean(x[-3:])
                        pressure=pid(x_pid)
                        locx.append(x_pt)
                        x.append(x_pt)
                        y.append(y_pt)
                        print(x_pt)
                        t.append(i)
                # print(drift)
                        if  ts > 0 and p>10 and  p<1499:
                            
                        #     p=p+2
                        #     fgt_set_pressure(0, p)
                        # elif drift>2 and drift<50 and ts > 0 and  p>100 and  p<1499:
                        #     p=p-3
                        #     fgt_set_pressure(0, p)
                            if p<press-50:
                                p=press-50
                        #     fgt_set_pressure(0, p)
                            elif p>press+50:
                                p=press+50
                            else:
                                p+=pressure
                        
                        else:
                            p=p

                    # print(time.time())
                        pressures.append(p)
                        fgt_set_pressure(0, p)
                        # x_old=x
                        # y_old=y
                        i=i+1
                    run=True
                else:
                    run==False
                cam.stop()
        print(n)
        if locx.count(0)<40:

            path=name#'track_vids/20250313_MGH2118/5/vids/20250313_MGH2118_21_'+str(n)+'.h5'
            # path2='track_vids/20250313_MGH2118/5/press/20250313_MGH2118_21_press_'+str(n)+'.h5'
            with h5py.File(path,'w') as h5f:
                h5f.create_dataset("data", data=imgs)
            # with h5py.File(path2,'w') as h5f:
            #     h5f.create_dataset("data", data=pressures)
            n=n+1
        else:
            n=n
