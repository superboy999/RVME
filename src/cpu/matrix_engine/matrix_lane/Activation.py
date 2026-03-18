'''
Author: superboy
Date: 2024-06-17 16:02:50
LastEditTime: 2024-06-17 16:02:50
LastEditors: superboy
Description: 
FilePath: /gem5-rvm/src/cpu/matrix_engine/matrix_lane/Activation.py

'''
from m5.params import *
from m5.objects.SimObject import SimObject

class Activation(SimObject):
    type = 'Activation'
    cxx_header = "cpu/matrix_engine/matrix_lane/Activation.hh"
    cxx_class = "gem5::Activation"

    mode = Param.Unsigned("Choose the activation layer mode")