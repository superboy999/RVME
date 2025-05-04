from m5.params import *
from m5.objects.TickedObject import TickedObject

class XYBuffer(TickedObject):
    type = 'XYBuffer'
    cxx_header = "cpu/matrix_engine/matrix_lane/XYBuffer.hh"
    cxx_class = "gem5::XYBuffer"

    buffer_depth = Param.Unsigned("XYBuffer depth")
    # compute_units = VectorParam.ComputeUnit("Compute Units Array")
    data_width = Param.Unsigned("XYBuffer data width")
    # data_size = Param.Unsigned("XYBuffer data size")
    XY = Param.Bool("Decide if it's XBuffer or YBuffer")
    num_port = Param.Unsigned("Number of the XYBuffer port")