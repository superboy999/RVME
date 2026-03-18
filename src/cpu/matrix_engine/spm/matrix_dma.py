from m5.params import *
from m5.objects.TickedObject import TickedObject
from m5.proxy import *

class MatrixDmaDevice(TickedObject):
    type = "MatrixDmaDevice"
    cxx_header = "cpu/matrix_engine/spm/matrix_dma.hh"
    cxx_class = "gem5::MatrixDmaDevice"
    dma_buffer_width = Param.Unsigned(32, "Width of the DMA buffer in SPM (in bytes)")
    dma_buffer_depth = Param.Unsigned(32, "Depth of the DMA buffer in SPM")
    dma_transpose_buffer_num = Param.Unsigned(4, "Number of transpose buffers in SPM")
    dma_transpose_buffer_width = Param.Unsigned(32, "Width of the transpose buffer in SPM (in bytes)")
    dma_transpose_buffer_depth = Param.Unsigned(8, "Depth of the transpose buffer in SPM")
    cache_line_size = Param.Unsigned(32, "Cache line size of the main memory (in bytes)")
    dmaPort = RequestPort("DMA port")
    # system = Param.System(Parent.any, "System this DMA is part of")
    # system = Param.System(Parent.system, "...")