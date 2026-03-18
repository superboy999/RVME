from m5.params import *
# from AbstractMemory import *
# from DRAMCtrl import *
# from SimpleMemory import *
from m5.objects.AbstractMemory import *
from m5.objects.SimpleMemory import *
from m5.objects.MemCtrl import *
from m5.objects.DRAMInterface import *

# BASED ON OWN IMPLEMENTATION (SimpleMemory)

# ORIGINAL Simple memory: 30ns and 0ns. Bandwith=12.8GB/s (DDR3-1600)
# class ScratchpadMemory(SimObject):
class MatrixSPM(SimpleMemory):
   type = 'MatrixSPM'
   cxx_header = "cpu/matrix_engine/spm/matrix_spm.hh"
   cxx_class = "gem5::MatrixSPM"
   # port = ResponsePort("This port sends responses and receives requests")
   latency_write = Param.Latency('10ns', "Write latency in SPM")
   latency_write_var = Param.Latency('0ns', "Write latency in SPM variable")
   latency = Param.Latency("0ns", "Request to response latency")
   latency_var = Param.Latency("0ns", "Request to response latency variance")
   # Modeling energy
   energy_read = Param.Float('300', "Energy for each reading (pJ)")
   energy_write = Param.Float('430', "Energy for each writting (pJ)")
   energy_overhead = Param.Float('100', "Overhead energy (pJ)")

   # This parameter is defined as the acceptance rate of request. Not very clear...
   # In the config.ini file is the inverse, e.g. BW=12.8GB/s, bandwidth = 73.0 ps/b
   #
   # In some papers it is described as the maximum transfer rate, description that makes sense
#    bandwidth = Param.MemoryBandwidth('64GB/s',
#                                      "Combined read and write bandwidth")
   sram_banks = Param.Unsigned(4, "Number of SRAM banks in SPM")
   entry_width = Param.Unsigned(8, "Width of each entry in SPM (in bytes)")
   entry_depthA = Param.Unsigned(8, "Depth of set A in SPM")
   entry_depthB = Param.Unsigned(8, "Depth of set B in SPM")
   entry_depthC = Param.Unsigned(8, "Depth of set C in SPM")
   rw_ports_per_bank = Param.Unsigned(0, "Number of ports per bank in SPM") 
   r_ports_per_bank = Param.Unsigned(1, "Number of ports per bank in SPM") # only read ports
   w_ports_per_bank = Param.Unsigned(1, "Number of ports per bank in SPM") # only write ports
   bankBufferSize = Param.Unsigned(4, "depth of the bank buffer in SPM")
   dma_buffer_width = Param.Unsigned(32, "Width of the DMA buffer in SPM (in bytes)")
   dma_buffer_depth = Param.Unsigned(8, "Depth of the DMA buffer in SPM")
   # mode = Param.Unsigned(1, "Mode of the SPM") # 0: normal, 1: A、B、C mode
   # memSidePort = RequestPort("SPM memory port")
   dmaDevice = Param.MatrixDmaDevice("DMA device for SPM access")
   # system = Param.System(Parent.any, "System this device is part of")
   lutEntries = Param.Unsigned(2048, "Number of entries in the LUT")