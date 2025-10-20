'''
Author: superboy
Date: 2025-10-13 19:20:57
LastEditTime: 2025-10-13 19:22:29
LastEditors: superboy
Description: 
FilePath: /gem5-rvm/src/cpu/matrix_engine/elementwise_unit/ew_unit.py

'''
from m5.params import *
from m5.objects.TickedObject import TickedObject

class ElementwiseUnit(TickedObject):
    type = 'ElementwiseUnit'
    cxx_header = "cpu/matrix_engine/elementwise_unit/ew_unit.hh"
    cxx_class = "gem5::ElementwiseUnit"
    parallel_ewu = Param.Unsigned(4, "Number of parallel Elementwise Units")
