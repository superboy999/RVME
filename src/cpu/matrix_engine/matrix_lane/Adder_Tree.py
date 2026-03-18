from m5.params import *
from m5.objects.TickedObject import TickedObject

class AdderTree(TickedObject):
    type = 'AdderTree'
    cxx_header = "cpu/matrix_engine/matrix_lane/Adder_Tree.hh"
    cxx_class = "gem5::AdderTree"

    # coordinatex = VectorParam.Unsigned([], "Adder Tree Array Coordinate 2D")
    coordinatex = Param.Unsigned("Adder Tree coordinatex")
    coordinatey = Param.Unsigned("Adder Tree coordinatey")
    adder_tree_row_size = Param.Unsigned("Adder Tree row size")
    adder_tree_column_size = Param.Unsigned("Adder Tree column size")
    en_activate = Param.Bool("Decide whether to use the Activation")
    quantization_layer = Param.Quantization("Quantization layer")
    activation_layer = Param.Activation("Activation layer")
    # zbuffer = Param.ZBuffer("ZBuffer in the Matrix Lane")