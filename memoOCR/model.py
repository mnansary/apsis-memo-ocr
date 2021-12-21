#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from os import name

#-------------------------
# imports
#-------------------------
import cv2
import math
from numpy.core.numeric import cross
import tensorflow as tf
from scipy.sparse import base
from .utils import *
import pandas as pd
import matplotlib.pyplot as plt
from .detector import CRAFT
from .robust_scanner import RobustScanner
import copy
#-------------------------
# class
#------------------------

class OCR(object):
    def __init__(self,model_dir):
        '''
            Instantiates an ocr model:
            args:
                model_dir               :   path of the model weights
            TODO:
                craft                   :   Pipeline
        '''
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # detector weight loading and initialization
        try:
            craft_weights=os.path.join(model_dir,'det',"memo.h5")
            self.craft=CRAFT(craft_weights)
                    
            LOG_INFO("Detector Loaded")    
        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")

        
        # recognizer weight loading and initialization
        try:
            self.rec=RobustScanner(model_dir,"rec")
            LOG_INFO("Recognizer Loaded")
        except Exception as e:
            LOG_INFO(f"EXECUTION EXCEPTION: {e}",mcolor="red")
        
        
            
    
    def extract(self,img,batch_size=32,debug=False,rf=5):
        '''
            predict based on datatype
            args:
                img                 :   image to infer on
                batch_size          :   batch size for inference
        '''    
        if type(img)==str:
            img=cv2.imread(img)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img,_=padDetectionImage(img)
        h,w,d=img.shape
        #img=cv2.resize(img,(rf*h,rf*w))
        img=remove_shadows(img)
        img_t=threshold_image(img,True)
        img[img_t==255]=(255,255,255)
        #img=cv2.merge((img,img,img))
        img=enhance_contrast(img)
        
        if debug:
            plt.imshow(img)
            plt.show()
        # detect
        boxes,crops=self.craft.detect(img,debug=debug)
        # recognize
        texts=self.rec.recognize(None,None,batch_size=batch_size,image_list=crops)

        df=pd.DataFrame({"box":boxes,"text":texts})
        return df,crops   
