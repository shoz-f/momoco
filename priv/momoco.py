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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from tf2onnx.convert import _convert_common
from tf2onnx import tf_loader, utils

from erlport.erlterms import Atom

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def load_onnx(path):
    return onnx.load(path.decode('utf-8'))

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def save_onnx(onnx, path):
    utils.save_protobuf(path.decode('utf-8'), onnx)
    #onnx.save(onnx, path.decode('utf-8'))
    return Atom(b'ok')

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def to_tensorflow(onnx, path):
    tf_rep = prepare(onnx)
    tf_rep.export_graph(path.decode('utf-8'))

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def to_tflite(path):
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    tflite = converter.convert()
    with open(path+".tflite", "wb") as f:
        f.write(tflite)

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def from_saved_model(path):
    graph_def, inputs, outputs, initialized_tables, tensors_to_rename = tf_loader.from_saved_model(path, None, None, return_initialized_tables=True, return_tensors_to_rename=True)

    with tf.device("/cpu:0"):
        model_proto, _ = _convert_common(
            graph_def,
            name=path,
            input_names=inputs,
            output_names=outputs,
            tensors_to_rename=tensors_to_rename,
            initialized_tables=initialized_tables)

    return model_proto

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def from_tflite(path):
    with tf.device("/cpu:0"):
        model_proto, _ = _convert_common(
            None,
            name=path,
            tflite_path=path,
            output_path="xxx.onnx")

    return model_proto

# momoco.py
