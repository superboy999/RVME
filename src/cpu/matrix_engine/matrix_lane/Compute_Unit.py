from m5.params import *
from m5.objects.TickedObject import TickedObject

class ComputeUnit(TickedObject):
    type = 'ComputeUnit'
    cxx_header = "cpu/matrix_engine/matrix_lane/Compute_Unit.hh"
    cxx_class = "gem5::ComputeUnit"

    # coordinatex = VectorParam.Unsigned([], "Compute Unit Array Coordinate 2D")
    coordinatex = Param.Unsigned("Compute Unit coordinatex")
    coordinatey = Param.Unsigned("Compute Unit coordinatey")
    en_activate = Param.Bool("Decide whether to use the Activation")
    quantization_layer = Param.Quantization("Quantization layer")
    activation_layer = Param.Activation("Activation layer")
    # zbuffer = Param.ZBuffer("ZBuffer in the Matrix Lane")