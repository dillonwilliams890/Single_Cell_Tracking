#%%
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
from __future__ import print_function 
import time
import math
import flir
import seaborn as sns
from simple_pid import PID
from Fluigent.SDK import fgt_init, fgt_close
from Fluigent.SDK import fgt_set_pressure, fgt_get_pressure, fgt_get_pressureRange
import tkinter as tk
#%%
core = Core()
print(core)
stage_device_label = core.get_xy_stage_device()
#%%
flir.main()
#%%

with Camera() as cam: # Initialize Camera
    cam.GainAuto = 'Off'
    # Set the gain to 20 dB or the maximum of the camera.
    gain = min(10, cam.get_info('Gain')['max'])
    print("Setting gain to %.1f dB" % gain)
    cam.Gain = gain
    cam.ExposureAuto = 'Off'
    cam.ExposureTime = 400 # microseconds


#%%
def beeg_Yoshi():
    # create child window
    koniec=tk.Tk()
    koniec.minsize(width=420, height=450)
    koniec.title("Vítaz!")
    canvas = tk.Canvas(koniec, width=420, height=420)
    canvas.pack(anchor=tk.CENTER)
    img=tk.PhotoImage(file='beeg.png')
    canvas.create_image(210,210,image=img)
    canvas.image = img
    tk.Button(koniec, text='Sorry Yoshi', command=koniec.destroy).pack()
    koniec.mainloop()

def saturation(b):
    mass=[]
    parea=6.5 #camera pixel area
    #Molecular absorbtion coefficints of something like that ~chemistry~
    w430_o = 2.1486*(10**8)
    w430_d = 5.2448*(10**8)
    w410_o = 4.6723*(10**8)
    w410_d = 3.1558*(10**8)
    for blue in b:
        f = tp.locate(blue, 41, invert=True, topn=1, max_iterations=1)
        xf=f.x.values[0]
        yf=f.y.values[0]
        x=int(xf)
        y=int(yf)
        # print(x)
        # tp.annotate(f, blue)
        signal=f.signal.values[0]
        # print(signal)
        if y>240 or y<50 or x>335 or x<25 or signal<30:
            mass=[1,1]
            break
        else:
            # print(y)
            
            Hb=(blue[int(y-25):int(y+25), int(x-25):int(x+25)])
            base=Hb[25,25]+30
            # print(Hb)
            kernel = np.ones((3,3),np.uint8)
            mask = cv.GaussianBlur(Hb, (5, 5), 0)
            mask=cv.erode(mask,kernel,iterations=1)
            _, mask = cv.threshold(mask, base, 255, cv.THRESH_BINARY)
            masked = cv.bitwise_and(Hb, mask)
            average = masked[np.nonzero(masked)].mean()
            Hbnorm=Hb/average
            Hbnorm[Hbnorm <= 0] = 0.01
            hbmass=((parea*(10**-8)*64500*np.sum(np.sum((-np.log10(Hbnorm))))))
            mass.append(hbmass)
            # print(hbmass)
            # plt.imshow(Hbnorm)
            gray = cv.cvtColor(blue, cv.COLOR_BayerBG2GRAY)
            cv.rectangle(gray, (x-20, y-20), (x+20, y+20), (255,100,200),2)
            cv.putText(gray, str('%f' %hbmass), (x, y + 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            cv.imshow('blue', gray)
            # # press 'q' to break loop and close window
            cv.waitKey(0)
    
    e=mass[1] #410
    f=mass[0] #430
    #Set absorbtion values to equation constants
    a=w410_d
    b=w410_o
    c=w430_d
    d=w430_o
                
    #Calcuate mass of oxygenated and deoxygenated hemoglobin
    Mo=(a*f-e*c)/(a*d-b*c)
    Md=(e*d-b*f)/(a*d-b*c)
    saturation = Mo/(Mo+Md)

    return saturation


def speed(img):
    cx=0
    ts=0
    # frame=img
    L1=img[0][1::2, 1::2]
    L2=img[1][1::2, 1::2]
    if np.mean(L1)>np.mean(L2):
        b=[L1,L2]
    else:
        b=[L2,L1]
    f = tp.locate(b[1], 41, preprocess=True, percentile=99,  invert=True, max_iterations=1, characterize =True, topn=1)
    if len(f)<1:
        signal=0
    elif len(f)>0:
        signal=f.signal.values[0]
        xf=f.x.values[0]

        x=int(xf)
        signal=f.signal.values[0]
    # y=int(yf)
    if signal>70:
        ts = time.time()
        cx=x
    else:
        cx=0
        ts=0
    return cx, ts
#%%
def initial(img):
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
    f = tp.locate(b[0], 41, preprocess=True, percentile=99,  invert=True, max_iterations=1, characterize =True, topn=2)
    if len(f)<1:
        signal=0
    elif len(f)>0:
        f=f.max()
        signal=f.signal
        x=f.x
    # y=f.y.values[0]
    # x=int(xf)
    # y=int(yf)
    
    print(signal)
    # print(x)
    # y=int(yf)
    if signal>45 and x<300:
        cell=True
    return cell

def feedback(frames,x_old, y_old):
    thresh=50
    img=frames
    L1=img[0][1::2, 1::2]
    L2=img[1][1::2, 1::2]
    if np.mean(L1)>np.mean(L2):
        b=[L1,L2]
    else:
        b=[L2,L1]
    f = tp.locate(b[0], 41, preprocess=True, percentile=99,  invert=True, max_iterations=1, characterize =True, topn=2)
    # gray = cv.cvtColor(b[0], cv.COLOR_BayerBG2GRAY)
    if len(f)<1:
        pt=[0,0,0]
        ts=0
    elif len(f)==1:
        pt=[f.x.values[0],f.y.values[0], f.signal.values[0]]
        ts = time.time()
    elif len(f)>1:
        pt1=[f.x.values[0],f.y.values[0], f.signal.values[0]]
        ts = time.time()
        pt2=[f.x.values[1],f.y.values[1], f.signal.values[1]]
        # print(pt1)
        # print(pt2)
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
            ts=0
    
    return pt, ts
 #%%
x_init=core.get_x_position()
y_init=core.get_y_position()
z_init=core.get_position()
# x_end=(x_init+500.0)
#%%
x_fin=x_init+1500
y_fin=y_init
z_fin=z_init
#%%
x_fin=core.get_x_position()
y_fin=core.get_y_position()
z_fin=core.get_position()
#%%
x_init=x_fin-10500
y_init=y_fin
z_init=z_fin
#%%
core.set_xy_position(x_init,y_init)
# x=core.get_x_position()
#%%
core.set_xy_position(x_fin,y_fin)
# %% 
core.set_xy_position(x_init,y_init)
core.wait_for_device(core.get_xy_stage_device())
fgt_init()
fgt_set_pressure(0, 150)
#%%
i=0
imgs=[]
with Camera() as cam: # Acquire and initialize Camera
    cam.start() # Start recording
    while i<200:
        if i<1:
            x_old=0
            ts_old=0
            # cam.start() # Start recording
            img = [cam.get_array() for n in range(2)] # Get 10 frames
            # cam.stop() # Stop recording
            # frame=np.maximum(img[0],img[1])
            frame = img#np.maximum(data2[i][0],data2[i][1])
            x, t = speed(frame)
            x_old=x; t_old=t
            i=i+1
        else:
            # cam.start() # Start recording
            img = [cam.get_array() for n in range(2)] # Get 10 frames
            # cam.stop() # Stop recording
            # frame=np.maximum(img[0],img[1])
            frame = img#np.maximum(data2[i][0],data2[i][1])
            # imgs.append(img)
            x, t = speed(frame)
            if x>0 and x_old>0:
                v=2*0.1725*(2*x-2*x_old)/(t-t_old)
            else:
                 v=0
            print(v)
            x_old=x; t_old=t
            i=i+1
    cam.stop() # Stop recording
    cv.destroyAllWindows()

#%%
press=150
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
pid = PID(0.1, 0.01, 1, setpoint=200)
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
        frame = [cam.get_array() for n in range(1)]
        cell=initial(frame)
        print(cell)
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
                pt, ts = feedback(img,x_old, y_old)
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
                    if p<press-20:
                        p=press-20
                #     fgt_set_pressure(0, p)
                    elif p>press+20:
                        p=press+20
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
plt.plot(locx)
# %% 
fgt_init()   
fgt_set_pressure(0, 215)
core.set_xy_position(x_init,y_init)
core.wait_for_device(core.get_xy_stage_device())


 #%%
# i=0
# p=400
# imgs=[]
# with Camera() as cam: # Acquire and initialize Camera
#     cam.start() # Start recording
#     fgt_init()
#     fgt_set_pressure(0, p)
#     x_old=0
#     y_old=0
#     core.set_xy_position(x_fin,y_fin)
#     while core.get_x_position()<x_fin:
#             # cam.start() # Start recording
#             img = [cam.get_array() for n in range(2)] # Get 10 frames
#             # cam.stop() # Stop recording
#             # frame=np.maximum(img[0],img[1])
#             frame = img#np.maximum(data2[i][0],data2[i][1])
#             pt, ts = feedback(frame,x_old, y_old)
#             x=pt[0]; y=pt[1]
#             # print(2*x)
#             imgs.append(img)
#             drift=2*0.1725*(x-x_old)
#             # print(drift)
#             if drift<-2 and drift>-50  and ts > 0 and p>200 and  p<599:
#                 p=p+2
#                 fgt_set_pressure(0, p)
#             elif drift>2 and drift<50 and ts > 0 and  p>200 and  p<599:
#                 p=p-2
#                 fgt_set_pressure(0, p)
#             elif p<200:
#                 p=300
#                 fgt_set_pressure(0, p)
#             elif p>599:
#                 p=400
#                 fgt_set_pressure(0, p)
#             else:
#                 p=p
#             # print(time.time())
#             x_old=x
#             y_old=y
#             i=i+1
#     cam.stop() # Stop recording        # print(x)
#     # cv.destroyAllWindows()
 # %%
 # %%
with h5py.File('track_vids/20241212_MGH2100/0/20241212_MGH2100_0_1.h5','w') as h5f:
    h5f.create_dataset("data", data=imgs)
# %%
img=imgs[10]
# gray = cv.cvtColor(frame, cv.COLOR_BayerBG2GRAY)
L1=img[0][1::2, 1::2]
L2=img[1][1::2, 1::2]
print(np.mean(L2))
# %%
i=0         
while i<150:
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
    f = tp.locate(b[1], 41, preprocess=True, percentile=99,  invert=True, max_iterations=1, characterize =True, topn=1)
    # t=f[f['signal']==f['signal'].max()]
    frame = cv.cvtColor(b[1], cv.COLOR_BayerBG2GRAY)
    # # print(signal)
    signal=f.signal.values[0]
    xf=f.x.values[0]
    yf=f.y.values[0]
    x=int(xf)
    y=int(yf)
    # if signal>14:
    # cv.rectangle(frame, (x-20, y-20), (x+20, y+20), (255,100,200),2)
    cv.putText(frame, str('%f' %signal), (x, y - 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        # drift=0.1725*(x-x_old)
    cv.imshow('Single Track', frame)
    # # press 'q' to bre
    cv.waitKey(5)
    i=i+1
cv.destroyAllWindows()
# %%
core.set_xy_position(x_init,y_init)
#%%
with Camera() as cam: # Acquire and initialize Camera
        core.set_xy_position(x_fin,y_fin) 
        cam.start()  
        while core.get_x_position()<x_fin-1:
            print(time.time())
            img = [cam.get_array() for n in range(2)] # Get 10 frames
            frame = cv.cvtColor(img[0], cv.COLOR_BayerBG2GRAY)[1::2, 1::2]
            cv.imshow('Single Track', frame)
            print(time.time())
        cam.stop()
cv.destroyAllWindows()
# %%

#%%
n=1
while n<11:
    press=240
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
    pid = PID(0.1, 0.01, 1, setpoint=200)
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
            cell=initial(frame)
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
                    pt, ts = feedback(img,x_old, y_old)
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
                        if p<press-20:
                            p=press-20
                    #     fgt_set_pressure(0, p)
                        elif p>press+20:
                            p=press+20
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

        path='track_vids/20241212_MGH2100/0_0/20241212_MGH2100_0_'+str(n)+'.h5'
        # path2='track_vids/20241205_CHC022/3_1500/pressures/20241120_CHC047_3_1500_pressure_'+str(n)+'.h5'
        with h5py.File(path,'w') as h5f:
            h5f.create_dataset("data", data=imgs)
        # with h5py.File(path2,'w') as h5f:
        #     h5f.create_dataset("data", data=pressures)
        n=n+1
    else:
        n=n

# beeg_Yoshi()
# %%
# window=tk.Tk()
# window.title('Hello Python')
# window.geometry("400x400")
# window.mainloop()
beeg_Yoshi()
# %%