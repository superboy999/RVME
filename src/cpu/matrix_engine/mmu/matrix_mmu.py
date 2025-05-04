from m5.params import *
from m5.objects.ClockedObject import ClockedObject
from m5.proxy import *
from m5.objects.BaseO3CPU import BaseO3CPU

class MatrixMMU(ClockedObject):
    type = 'MatrixMMU'
    cxx_header = "cpu/matrix_engine/mmu/matrix_mmu.hh"
    cxx_class = "gem5::MatrixMMU"
    system = Param.System(Parent.any, "System this tester is part of")
    num_matrix_mem_ports = Param.Unsigned("Number of Matrix Memory Ports")
    w_channel = Param.Unsigned(0, "Matrix Memory Write Channel")
    r_channel = Param.Unsigned(1, "Matrix Memory Read Channel")
    pending_depth = Param.Unsigned("Matrix MMU Pending Queue depth")
    matrix_mem_ports = VectorRequestPort("Matrix Accelerator Memory Port")
    # o3cpu = Param.BaseO3CPU("Host CPU O3 ")