#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function

#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import cv2 
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt 
import random
import base64
from io import BytesIO

#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path
#---------------------------------------------------------------
# image utils
#---------------------------------------------------------------
def stripPads(arr,
              val):
    '''
        strip specific value
        args:
            arr :   the numpy array (2d)
            val :   the value to strip
        returns:
            the clean array
    '''
    # x-axis
    arr=arr[~np.all(arr == val, axis=1)]
    # y-axis
    arr=arr[:, ~np.all(arr == val, axis=0)]
    return arr
#---------------------------------------------------------------
def locateData(img,val):
    '''
        locates img data based on specific value threshold
    '''
    idx=np.where(img>val)
    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    return y_min,y_max,x_min,x_max
#---------------------------------------------------------------
def remove_shadows(image: np.ndarray):
    # split the image channels into b,g,r
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    rgb_planes = [b,g,r]

    # iniatialising the final shadow free normalised image list for planes
    result_norm_planes = []

    # removing the shadows in individual planes
    for plane in rgb_planes:
        # dialting the image to spead the text to the background
        dilated_image = cv2.dilate(plane, np.ones((7,7), np.uint8))
        
        # blurring the image to get the backround image
        bg_image = cv2.medianBlur(dilated_image, 21)

        # subtracting the plane-background from the image-plane
        diff_image = 255 - cv2.absdiff(plane, bg_image)

        # normalisng the plane
        norm_image = cv2.normalize(diff_image,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # appending the plane to the final planes list
        result_norm_planes.append(norm_image)

    # merging the shadow-free planes into one image
    normalised_image = cv2.merge(result_norm_planes)

    # returning the normalised image
    return normalised_image
#---------------------------------------------------------------
def threshold_image(img,blur):
    '''
        threshold an image
    '''
    assert len(img.shape)==3
    # grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold
    if blur:
        img = cv2.GaussianBlur(img,(5,5),0)
    _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img
#---------------------------------------------------------------
# detection utils
#---------------------------------------------------------------
def padDetectionImage(img):
    cfg={}
        
    h,w,d=img.shape
    if h>w:
        # pad widths
        pad_width =h-w
        # pads
        pad =np.ones((h,pad_width,d))*255
        # pad
        img =np.concatenate([img,pad],axis=1)
        # cfg
        cfg["pad"]="width"
        cfg["dim"]=w
    
    elif w>h:
        # pad height
        pad_height =w-h
        # pads
        pad =np.ones((pad_height,w,d))*255
        # pad
        img =np.concatenate([img,pad],axis=0)
        # cfg
        cfg["pad"]="height"
        cfg["dim"]=h
    else:
        cfg=None
    return img.astype("uint8"),cfg

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 100 and (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_rotate_crop_image(img, points):
    # Use Green's theory to judge clockwise or counterclockwise
    # author: biyanhua
    d = 0.0
    for index in range(-1, 3):
        d += -0.5 * (points[index + 1][1] + points[index][1]) * (
                    points[index + 1][0] - points[index][0])
    if d < 0: # counterclockwise
        tmp = np.array(points)
        points[1], points[3] = tmp[3], tmp[1]

    try:
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
    except Exception as e:
        print(e)
#---------------------------------------------------------------
# recognition utils
#---------------------------------------------------------------
def padData(img,pad_loc,pad_dim,pad_type,pad_val):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto
            pad_type:       central or left aligned pad
            pad_val :       the value to pad 
    '''
    
    if pad_loc=="lr":
        # shape
        h,w,d=img.shape
        if pad_type=="central":
            # pad widths
            left_pad_width =(pad_dim-w)//2
            # print(left_pad_width)
            right_pad_width=pad_dim-w-left_pad_width
            # pads
            left_pad =np.ones((h,left_pad_width,3))*pad_val
            right_pad=np.ones((h,right_pad_width,3))*pad_val
            # pad
            img =np.concatenate([left_pad,img,right_pad],axis=1)
        else:
            # pad widths
            pad_width =pad_dim-w
            # pads
            pad =np.ones((h,pad_width,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=1)
    else:
        # shape
        h,w,d=img.shape
        # pad heights
        if h>= pad_dim:
            return img 
        else:
            pad_height =pad_dim-h
            # pads
            pad =np.ones((pad_height,w,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=0)
    return img.astype("uint8") 
#---------------------------------------------------------------
def padWords(img,dim,ptype="central",pvalue=255,scope_pad=50):
    '''
        corrects an image padding 
        args:
            img     :       numpy array of single channel image
            dim     :       tuple of desired img_height,img_width
            ptype   :       type of padding (central,left)
            pvalue  :       the value to pad
        returns:
            correctly padded image

    '''
    img_height,img_width=dim
    mask=0
    # check for pad
    h,w,d=img.shape
    w_new=int(img_height* w/h) 
    img=cv2.resize(img,(w_new,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    h,w,d=img.shape
    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w) 
        img=cv2.resize(img,(img_width,h_new),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # pad
        img=padData(img,
                     pad_loc="tb",
                     pad_dim=img_height,
                     pad_type=ptype,
                     pad_val=pvalue)
        mask=img_width

    elif w < img_width:
        # pad
        img=padData(img,
                    pad_loc="lr",
                    pad_dim=img_width,
                    pad_type=ptype,
                    pad_val=pvalue)
        mask=w+scope_pad
        if mask>img_width:
            mask=img_width
    
    # error avoid
    img=cv2.resize(img,(img_width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img,mask 
#---------------------------------------------------------------
#------------------------------------
# region-utils 
#-------------------------------------
def intersection(boxA, boxB):
    # boxA=ref
    # boxB=sig
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    x_min,y_min,x_max,y_max=boxB
    selfArea  = abs((y_max-y_min)*(x_max-x_min))
    return interArea/selfArea
#---------------------------------------------------------------
def localize_box(box,region_boxes):
    '''
        lambda localization
    '''
    max_ival=0
    box_id=None
    for idx,region_box in enumerate(region_boxes):
        ival=intersection(region_box,box)
        if ival==1:
            return idx
        if ival>max_ival:
            max_ival=ival
            box_id=idx
    if max_ival==0:
        return None
    return box_id

#----------------------------------------
# display utils
#----------------------------------------
def display_data(info,img,cv_color=True):
    if type(img)=="str":
        img=cv2.imread(img)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    else:
        if cv_color:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    LOG_INFO("------------------------------------------------")
    LOG_INFO(f"{info}")
    LOG_INFO("------------------------------------------------")
    plt.imshow(img)
    plt.show()

#-----------------------
# response utils
#-----------------------

def img2base64(img_detections):
    '''
        creates proper json response
    '''
    # Convert numpy array To PIL image
    pil_detection_img = Image.fromarray(img_detections)

    # Convert PIL image to bytes
    buffered_detection = BytesIO()

    # Save Buffered Bytes
    pil_detection_img.save(buffered_detection, format='PNG')

    # Base 64 encode bytes data
    # result : bytes
    base64_detection = base64.b64encode(buffered_detection.getvalue())

    # Decode this bytes to text
    # result : string (utf-8)
    base64_detection = base64_detection.decode('utf-8')
    return base64_detection

