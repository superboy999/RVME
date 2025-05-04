from m5.params import *
from m5.objects.ClockedObject import ClockedObject

class MatrixRF(ClockedObject):
    type = 'MatrixRF'
    cxx_header = "cpu/matrix_engine/matrix_registerfile/matrix_reg.hh"
    cxx_class = "gem5::MatrixRF"

    physicReg_num = Param.Unsigned(16, "Physical Register Number")
    regWidth = Param.Unsigned("Register data width")
    bank_num = Param.Unsigned("Register Bank Number")
    bank_depth = Param.Unsigned("Register Bank Depth")