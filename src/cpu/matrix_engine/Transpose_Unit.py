from m5.params import *
from m5.objects.TickedObject import TickedObject

class TransposeUnit(TickedObject):
    type = 'TransposeUnit'
    cxx_header = "cpu/matrix_engine/transpose_unit/Transpose_Unit.hh"
    cxx_class = "gem5::TransposeUnit"

    buffer_depth = Param.Unsigned("TransposeUnit depth")
    # compute_units = VectorParam.ComputeUnit("Compute Units Array")
    # data_width = Param.Unsigned("TransposeUnit data width")
    # data_size = Param.Unsigned("TransposeUnit data size")
    num_port = Param.Unsigned("Number of the TransposeUnit port")