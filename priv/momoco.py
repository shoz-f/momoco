#!/usr/local/bin/python
# -*- coding: utf-8 -*-
################################################################################
# momoco.py
# Description:  
#
# Author:       shozo fukuda
# Date:         Thu Jun 30 22:24:53 2022
# Last revised: $Date$
# Application:  Python 3
################################################################################

#<IMPORT>
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

#tf_rep = prepare(onnx_model)  # prepare tf representation
#tf_rep.export_graph("output_path")  # export the model

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def load_onnx(path):
    return onnx.load(path)

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def to_tensorflow(onnx, path):
    tf_rep = prepare(onnx)
    tf_rep.export_graph(path)

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def to_tflite(onnx, path):
    tf_rep = prepare(onnx)
    tf_rep.export_graph(path)

    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    tflite = converter.convert()
    with open(path.decode('utf-8')+".tflite", "wb") as f:
        f.write(tflite)

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def get_tflite(path):
    converter = tf.lite.TFLiteConverter.from_saved_model(path.decode('utf-8'))
    tflite = converter.convert()
    with open("log.txt", "a") as f:
        print(sys.stdout, file=f)
        print(sys.__stdout__, file=f)
    return "ok"

def dummy():
    os.dup2(2, 1)
    return "ok"

# momoco.py
