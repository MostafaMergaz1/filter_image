import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import cv2

st.title(' filterimage')

uploader=st.file_uploader('choose image',type=['jpg','png','jpeg'])

def convert1(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

def convert2(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2Lab)

def convert3(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2HSV)


def vintage (img,level):
    if level==0:
        level=1
    hight,width=img.shape[:2]
    x_kernel=cv2.getGaussianKernel(width,width/level)
    y_kernel=cv2.getGaussianKernel(hight,hight/level)
    kernel=y_kernel*x_kernel.T
    mask=kernel/kernel.max()
    img_copy=np.copy(img)
    for  i in range(3):
        img_copy[:,:,i]=img_copy[:,:,i]*mask
    return img_copy


def HDR(img,ksize,sigma_s,sigma_r):
    convert=cv2.convertScaleAbs(img,beta=ksize)
    detail=cv2.detailEnhance(convert,sigma_s=sigma_s,sigma_r=sigma_r)
    return detail


def bluring(img,ksize):
    ksize=abs(int(ksize))
    if ksize%2==0:
        ksize+=1
    return cv2.GaussianBlur(img,(ksize,ksize),0,0)


def dest_gray(img,ksize):
    ksize=abs(int(ksize))
    if ksize%2==0:
        ksize+=1
    blur=cv2.GaussianBlur(img,(ksize,ksize),0,0)
    dst_gray,dst_color= cv2.pencilSketch(blur)
    return dst_gray

def color_gray(img,ksize):
    ksize=abs(int(ksize))
    if ksize%2==0:
        ksize+=1
    blur=cv2.GaussianBlur(img,(ksize,ksize),0,0)
    dst_gray,dst_color=cv2.pencilSketch(blur)
    return dst_color
def style_img(img,ksize,sigma_s,sigma_r):
    blur=cv2.GaussianBlur(img,(ksize,ksize),0,0)
    return cv2.stylization(blur,sigma_s=sigma_s,sigma_r=sigma_r)



def naration_img(img):
    return 255-img

def sketch_effect(img,ksize):
    ksize=abs(int(ksize))
    if ksize%2==0:
        ksize+=1
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (ksize, ksize), 0,0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return sketch
    
def enhance_old_image(img):
    # 1. إزالة التشويش
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # 2. ضبط التباين والسطوع
    contrast_bright = cv2.convertScaleAbs(denoised, alpha=1.3, beta=20)

    # 3. زيادة الحدة
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(contrast_bright, -1, kernel)

    # 4. تحسين التفاصيل
    enhanced = cv2.detailEnhance(sharpened, sigma_s=10, sigma_r=0.15)

    return enhanced

if uploader is not None:
    img=Image.open(uploader)
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    original_image,output_image=st.columns(2)

    with original_image:
        st.header('Original Image')
        st.image(img,channels='BGR',use_column_width=True)
        st.header('choose from filtter')
        optoin=st.selectbox('silict from filtter',('None','convert1','convert2','convert3','vintage','bluring','HDR','dest_gray','naration_img','color_gray','style_img','sketch_effect','enhance_old_image'
))
    output_flag=1
    color='BGR'

    if optoin=='None':
        output_flag=0
        output=img
        
    elif optoin=='convert1':
        output=convert1(img)
        color='GRAY'

    elif optoin=='convert2':
        output=convert2(img)

    elif optoin=='convert3':
        output=convert3(img)

    elif optoin=='vintage':
        level=st.slider('level',-20,20,1,step=1)
        output=vintage(img,level)

    elif optoin=='bluring':
        ksize=st.slider('ksize',-20,20,0,step=1)
        output=bluring(img,ksize)

    elif optoin=='HDR':
        bet=st.slider('bets',-100,100,0,step=1)
        sigma_s=st.slider('sigmaz-s',0,100,0,step=1)
        sigma_r = st.slider('sigmaz-r', 0.0, 1.0, 0.5, step=0.01)

        output=HDR(img,bet,sigma_s,sigma_r)

    elif optoin=='dest_gray':
        ksize=st.slider('ksize',0,20,0,step=1)
        output=dest_gray(img,ksize)


    elif optoin=='color_gray':
        ksize=st.slider('ksize',0,20,0,step=1)
        output=color_gray(img,ksize)


    elif optoin=='style_img':
        bet=st.slider('bets',-100,100,0,step=1)
        sigma_s=st.slider('sigmaz-s',0,100,0,step=1)
        sigma_r=st.slider('sigmaz-r',0,5,0,step=1)
        output=style_img(img,bet,sigma_s,sigma_r)

    elif optoin == 'naration_img':
            output = naration_img(img)

    elif optoin == 'enhance_old_image':
            output = enhance_old_image(img)


    elif optoin == 'sketch_effect':
            ksize=st.slider('ksize',-300,0,300,step=1)
            output = sketch_effect(img,ksize)
            color = 'GRAY'


    with output_image:
        st.header('Output Image')
        if color=='BGR':
            output=cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
            st.image(output,use_column_width=True)
        elif color=='GRAY':
            st.image(output, channels='GRAY', use_column_width=True)
    
        
        
