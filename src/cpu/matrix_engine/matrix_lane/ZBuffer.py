from m5.params import *
from m5.objects.TickedObject import TickedObject

class ZBuffer(TickedObject):
    type = 'ZBuffer'
    cxx_header = "cpu/matrix_engine/matrix_lane/ZBuffer.hh"
    cxx_class = "gem5::ZBuffer"

    buffer_depth = Param.Unsigned("ZBuffer depth")
    # compute_units = VectorParam.ComputeUnit("Compute Units Array")
    data_width = Param.Unsigned("ZBuffer data width")
    # data_size = Param.Unsigned("ZBuffer data size")
    num_port = Param.Unsigned("Number of the XYBuffer port")
    # matrix_lane = Param.MatrixLane("Matrix Lane TOP")