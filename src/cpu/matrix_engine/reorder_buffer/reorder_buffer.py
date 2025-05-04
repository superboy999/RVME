from m5.params import *
from m5.objects.TickedObject import TickedObject

class ReorderBuffer(TickedObject):
    type = 'ReorderBuffer'
    cxx_header = "cpu/matrix_engine/reorder_buffer/reorder_buffer.hh"
    cxx_class = "gem5::ReorderBuffer"

    ROB_depth = Param.Unsigned(8, "Depth of the Reorder buffer")