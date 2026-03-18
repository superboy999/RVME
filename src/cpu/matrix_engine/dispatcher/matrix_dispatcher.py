from m5.params import *
from m5.objects.TickedObject import TickedObject

class MatrixDispatcher(TickedObject):
    type = 'MatrixDispatcher'
    cxx_header = "cpu/matrix_engine/dispatcher/matrix_dispatcher.hh"
    cxx_class = "gem5::MatrixDispatcher"

    accReg_num = Param.Unsigned(16, "Accumulator Register Number")
    tileReg_num = Param.Unsigned(16, "Tile Register Number")
    MQ_depth = Param.Unsigned(10, "Depth of the Memory Queue")
    AQ_depth = Param.Unsigned(10, "Depth of the Arithmetic Queue")
    # matrix_engine = Param.MatrixEngine("Matrix Engine TOP")