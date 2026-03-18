'''
Author: superboy
Date: 2025-09-30 23:57:03
LastEditTime: 2025-09-30 23:59:50
LastEditors: superboy
Description: 
FilePath: /gem5-rvm/src/cpu/matrix_engine/inst_buf/inst_buf.py

'''
from m5.params import *
from m5.objects.TickedObject import TickedObject

class InstructionBuffer(TickedObject):
    type = 'InstructionBuffer'
    cxx_header = "cpu/matrix_engine/inst_buf/inst_buf.hh"
    cxx_class = "gem5::InstructionBuffer"

    IB_depth = Param.Unsigned(128, "Depth of the Instruction buffer")