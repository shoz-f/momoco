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
from onnx2torch import convert

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def to_str(s):
    return s.decode('utf-8') if isinstance(s, bytes) else s

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def load_onnx(path):
    path = to_str(path)

    return onnx.load(path)

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def save_onnx(model, path):
    path = to_str(path)

    utils.save_protobuf(path, model)
    #onnx.save(onnx, path.decode('utf-8'))

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def to_tensorflow(model, path):
    path = to_str(path)

    tf_rep = prepare(model)
    tf_rep.export_graph(path)

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def to_tflite(path):
    path = to_str(path)

    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    tflite = converter.convert()
    with open(path+".tflite", "wb") as f:
        f.write(tflite)


#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def to_torch(onnx, path):
    model = convert(onnx)
    torch.jit.save(model, path)
    return model

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def from_saved_model(path):
    path = to_str(path)

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
    path = to_str(path)

    with tf.device("/cpu:0"):
        model_proto, _ = _convert_common(
            None,
            name=path,
            tflite_path=path)

    return model_proto

#<SUBROUTINE>###################################################################
# Function:     
# Description:  
# Dependencies: 
################################################################################
def remove_initializer_from_input(model, path):
    inputs = model.graph.input

    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, path)

# momoco.py
