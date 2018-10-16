# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 18:05:55 2018

@author: Priyanka Iyer
"""

from scipy import stats
import scipy
import numpy as np
from scipy import ndimage
import math as mt
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import glob
from skimage import io, exposure,filters,restoration,feature,measure
from os import path
import pandas as pd

a=1.5 ##occultor positions in R0
b=4

def parabola(x,x1,y1,a):
        return (x-x1)**2/(4*a) + y1
    
def process_image(pic):
    '''
    PROCESSES IMAGE , APPLIES GAMMA FILTER TO ENHANCE BRIGHTER REGIONS 
    '''
    pic=filters.median(pic)
    thresh=pic.max()/3
    (n,m)=pic.shape
    numx,numy=np.where(pic>thresh)
    if numx.size<200:
        pic1=exposure.adjust_gamma(pic, gamma=0.3, gain=1)
    elif numx.size>10000:
        pic1=exposure.adjust_gamma(pic, gamma=3, gain=1)
    else:
        pic1=pic
    return pic1

def avg_parabola_parameters2(img,cadence,A,S,K2,show_plot):
    '''
    RETURNS Average parabola based on centroid of region
    A= array with parabola parameter A
    S= accumulator array from HT
    K2 adaptive threshold for peak detection
    '''
    Accf = ndimage.gaussian_filter(S, 1)
    (n,m)=img.shape
    R0,y0,t0=image_values(img,cadence)
    thresh=np.mean(Accf)+np.std(Accf)*K2 ##Adaptive thsholding to detect connected regions
    rel_thresh=thresh/Accf.max()
    labeled, nr_objects = ndimage.label(Accf > rel_thresh*S.max()) 
    props=measure.regionprops(labeled)
    x_list=[]
    a_list=[]
    intensity=[]
    std_x=[]
    std_A=[]
    num_avg=[]
    for region in props:
        a,b=region.centroid
        x=list(region.coords[:,0])
        y=list(region.coords[:,1])
        if a>=0 and region.area>10: ###takes only averaged parabolas(not isolated ones)
            Stemp=S[x,y] 
            thresh=Stemp.max()*0.95 ##locates points with near maximum values in the given region
            idx=np.where(Stemp>thresh)
            idx=list(idx)
            for i in idx:
                idx=list(i)
            x=[x[i] for i in idx] ##takes co-ords of points with maximum intensity
            y=[y[i] for i in idx]
            wgt=Stemp[idx]
            x_list.append(np.average(x,weights=wgt)) ##weighted average of located points
            a_list.append(np.average(y,weights=wgt))##(more intensity points get more preference)
            mean_I=np.mean(wgt) ##mean intensity
            num_avg.append(len(x))
            intensity.append(mean_I)
            m1,s1 = stats.norm.fit(x)
            acc = [(1/(2*A[a]))*(y0/(t0*t0))for a in y] 
            m2,s2 = stats.norm.fit(acc)##mean and std deviation of acceleration
            std_x.append(s1)
            std_A.append(s2)
    return x_list,a_list,intensity,std_x,std_A,num_avg

def avg_vel(y,t):
    m,c=np.polyfit(t,y,deg=1)
    return m
    
def image_values(im,cadence):
    (n,m)=im.shape
    R0=6.955*(10**8)##m
    y0=(b-a)*R0/(n-21)
    t0=cadence*60/1 ##seconds
    return R0,y0,t0

def values_of_Acce(err):
    ###creates array for Acceleration based on relative error 
    A=500
    Acce=np.zeros((A,1))
    f=0
    while A>0.5:
        Atemp=A-A*err
        Acce[f]=Atemp
        f=f+1
        A=Atemp
    Acce1=[x for x in Acce if x>0]
    Acce1=np.asarray(Acce1)
    return Acce1[:,0]
    
def Parabolic_HT2(img_11,angle=0,image_p=0,time=0,show_plot=True,cadence=5 ,adap_thresh1=6.3,adap_thresh2=2.2):
    '''
    time is start time of input image
    To not display plot set show_plot=False
    To save figure set image_p=!0
    adap_thresh1 corresponds to image points to be considered in HT
    adap_thresh2 corresponds to threshold for peak detection 
    default values set for running difference images based on trial and error
    '''
    print('Parabolas are of the form (x-x1)^2=4*A*(y-y1) where y1 is fixed at R0')
    #########PLOTTING INPUT################
    img_1=process_image(img_11)
    fig=plt.figure()
    img_2=np.rot90(img_1,2) ##rotates image by 180
    img=np.fliplr(img_2) ##flips image left to right
    (n,m)=img.shape
    R0,y0,t0=image_values(img,cadence)
    ###################LOCATES START POINT Y1######3333
    y1=int((a-1)*R0/y0)
    y1=21-y1  ###sets start of cme at Y1=R0
    del_y=(b-1)/(n-1-y1)
    if show_plot==True:
        plt.imshow(img,cmap="gray",origin='origin',extent=[time, time+(m-1)*cadence/60, a-21*del_y, b],aspect='auto')
        plt.title('INPUT IMAGE')
        plt.xlabel('time(hours)')
        plt.ylabel('Distance in R0(From centre of the sun)')
        plt.show()
    if image_p!=0:
        fig.savefig(path.join('HTP/',"parabolicHT_0{0}.png".format(image_p)))
    #######HOUGH TRANSFORM###########
    thresh=np.mean(img)+np.std(img)*adap_thresh1
    y,x=np.where(img>thresh)
    x1=np.arange(0,m)
    #########GENERATES A###########
    Acce=values_of_Acce(0.05)
    A=(1/(2*Acce))*(y0/(t0*t0))
    ##############################
    Acc=np.zeros((len(x1),len(A)))
    for i in range(0,len(x)):
        for j in range(0,len(A)):
            xtemp=x[i]-mt.sqrt(4*A[j]*(y[i]-y1))
            if 0<=xtemp<=x1.max():
                idx = (np.abs(x1 - xtemp)).argmin()
                Acc[idx,j]+=1
    #####PLOTTING HOUGH SPACE########
    if show_plot==True:
        plt.imshow(Acc,cmap="gray",aspect='auto',origin='origin',extent= [A.min(),A.max(),time,time+(m-1)*cadence/60])
        plt.title('HOUGH SPACE')
        plt.xlabel('a(parabola parameter)')
        plt.ylabel('X1 parameter')
        plt.show()
    #######FINDS PEAKS######
    x_list,a_list,intensity,std_x,std_A,num_avg=avg_parabola_parameters2(img, cadence,A,Acc,adap_thresh2,show_plot)
    a_list=[int(round(a)) for a in a_list]
    acc=A[a_list]
    acc = [(1/(2*a))*y0/(t0*t0) for a in acc] ###VALUES OF ACCLERATION####
    fig=plt.figure()
    V_avg=[]
    x1=np.arange(0,m,0.05)
    x_list1=[time+x*cadence/60 for x in x_list]##list with transformedco-ordinates
########################################
    for i in range(0,len(x_list)):
        idx=(np.abs(x1 - x_list[i])).argmin()
        idx1=(np.abs(A - A[(a_list[i])])).argmin()
        x11=x1[idx:len(x1)-1]
        yplot=parabola(x11,x1[idx],y1,A[idx1])
        yplot=a-21*del_y+yplot*del_y
        yplot=[y for y in yplot if y<b]
        x11=time+x11*cadence/60
        x11=x11[0:len(yplot)]
        v_avg=avg_vel(yplot,x11)*R0/(1000*60*60) ##in km/s
        V_avg.append(v_avg)
        if show_plot==True:
            plt.imshow(img,cmap="gray",origin='origin',extent=[time, time+(m-1)*cadence/60, a-21*del_y, b],aspect='auto')
            plt.plot(x11,yplot,label='I:'+ str(intensity[i])+" Acc:"+str(acc[i])+" V_avg:"+str(V_avg[i]))
            plt.legend(loc='upper left')##PLOTS DETECTED PARABOLAS
            plt.title('DETECTED PARABOLAS')
            plt.xlabel('time(hours)')
            plt.ylabel('Distance in R0(From centre of the sun)')
            plt.ylim((1,b)) #limits plot to 1st quad
            plt.xlim((time,time+(m-1)*cadence/60)) 
    plt.show()
    #########CALCULATES ACCELERATION########      
    angle_list=[angle]*len(x_list1)
    if image_p!=0:
        fig.savefig(path.join('HTP/',"parabolicHT_{0}.png".format(image_p)))
    d = {'Angle':angle_list,'X-Coor':x_list1,'Acceleration':acc, 'Intensity':intensity,'V_avg':V_avg,'Std_A':std_A}
    df=pd.DataFrame(d)
    return(df)
    
'''
returns data frame containing x_values,intensity of parabola in hough space,std deviation in acceleration, acceleration,number of parabolas averaged and avg velocity of CME
'''