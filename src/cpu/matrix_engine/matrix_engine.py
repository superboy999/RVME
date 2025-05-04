from m5.params import *
from m5.SimObject import SimObject
import m5.defines

class MatrixEngine(SimObject):
    type = "MatrixEngine"
    cxx_header = "cpu/matrix_engine/matrix_engine.hh"
    cxx_class = "gem5::MatrixEngine"

    matrix_lanes = VectorParam.MatrixLane("Matrix Lanes in Matrix Engine")
    matrix_rename = Param.MatrixRename("Matrix Rename")
    matrix_rob = Param.ReorderBuffer("Matrix Reorder Buffer")
    lane_num = Param.Unsigned("Matrix Lane number")
    matrix_dispatcher = Param.MatrixDispatcher("Matrix Dispatcher in Matrix Engine")
    matrix_reg = Param.MatrixRF("Matrix Register File in Matrix Engine")
    matrix_mmu = Param.MatrixMMU("Matrix MMU in Matrix Engine")