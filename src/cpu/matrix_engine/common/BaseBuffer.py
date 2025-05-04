from m5.params import *

from m5.objects.ClockedObject import ClockedObject

class BaseBuffer(ClockedObject):
    type = 'BaseBuffer'
    cxx_header = 'cpu/matrix_engine/common/BaseBuffer.hh'

    input_size = Param.Unsigned(512, "input size of BaseBuffer")
    output_size = Param.Unsigned(512, "output size of BaseBuffer")
    data_width = Param.Unsigned(512, "data width in BaseBuffer")
    buffer_depth = Param.Unsigned(4, "data depth of BaseBuffer")