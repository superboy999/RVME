'''
Author: superboy
Date: 2024-06-17 16:45:21
LastEditTime: 2024-06-19 14:58:17
LastEditors: superboy
Description: 
FilePath: /gem5-rvm/src/cpu/matrix_engine/matrix_lane/Quantization.py

'''
from m5.params import *
from m5.objects.SimObject import SimObject

class Quantization(SimObject):
    type = 'Quantization'
    cxx_header = "cpu/matrix_engine/matrix_lane/Quantization.hh"
    cxx_class = "gem5::Quantization"

    scale_factor = Param.Float("Choose the scale factor")