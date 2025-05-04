from m5.params import *
from m5.objects.SimObject import SimObject

class MatrixRename(SimObject):
    type = 'MatrixRename'
    cxx_header = "cpu/matrix_engine/rename/matrix_rename.hh"
    cxx_class = "gem5::MatrixRename"

    numPhysicalRegs = Param.Unsigned(12, "Number of Matrix Physical Registers")
    numLogicalRegs = Param.Unsigned(8, "Number of Matrix Logical Registers")