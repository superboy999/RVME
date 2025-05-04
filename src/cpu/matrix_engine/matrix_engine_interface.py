from m5.params import *
from m5.SimObject import SimObject

class MatrixEngineInterface(SimObject):
    type = 'MatrixEngineInterface'
    cxx_header = "cpu/matrix_engine/matrix_engine_interface.hh"
    cxx_class = "gem5::MatrixEngineInterface"

    matrix_engine = Param.MatrixEngine("Matrix Engine In Interface")
    # matrix_dispatcher = Param.MatrixDispatcher("Matrix Dispatcher In Interface")
