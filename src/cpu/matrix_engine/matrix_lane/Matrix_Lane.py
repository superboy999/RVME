from m5.params import *
from m5.objects.TickedObject import TickedObject

class MatrixLane(TickedObject):
    type = 'MatrixLane'
    cxx_header = "cpu/matrix_engine/matrix_lane/Matrix_Lane.hh"
    cxx_class = "gem5::MatrixLane"

    XBuffer = Param.XYBuffer("XBuffer in Matrix Lane")
    YBuffer = Param.XYBuffer("YBuffer in Matrix Lane")
    zbuffer = Param.ZBuffer("ZBuffer in Matrix Lane")

    compute_units = VectorParam.ComputeUnit("compute units array")
    xbuffer_ports_num = Param.Unsigned("XBuffer Port Number")
    ybuffer_ports_num = Param.Unsigned("YBuffer Port Number")
    cu_row_size = Param.Unsigned("Compute Unit Row Size")
    cu_column_size = Param.Unsigned("Compute Unit Column Size")
    # matrix_reg = Param.MatrixRF("Matrix Register File")
    # matrix_engine = Param.MatrixEngine("Matrix Engine")
    WIDEN = Param.Unsigned(4, "int8 widen coeffience")