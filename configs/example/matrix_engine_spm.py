import sys
import math
import datetime

import m5
from m5.objects import *
from m5.util import addToPath, fatal
from optparse import OptionParser

addToPath('../')

from common import Options
from common import Simulation
from common import CacheConfig
from common import CpuConfig
from common import MemConfig
from common.Caches import *

# Parse options
ps = OptionParser()

# GENERAL OPTIONS
ps.add_option('--cmd',              type="string",
                                    help="Command to run on the CPU")
ps.add_option('--numThreads',       type="int", default=1,
                                    help="Number of threads running on the CPU")
ps.add_option('--output',           type="string",
                                    help="Outputs")
ps.add_option('--options',          type="string",
                                    help="Options")
ps.add_option('--cache_line_size',  type="int", default=64,
                                    help="System Cache Line Size in Bytes")
ps.add_option('--l1i_size',         type="string", default='32kB',
                                    help="L1 instruction cache size")
ps.add_option('--l1d_size',         type="string", default='32kB',
                                    help="L1 data cache size")
ps.add_option('--l2_size',          type="string", default='256kB',
                                    help="Unified L2 cache size")
ps.add_option('--mem_size',         type="string", default='2048MB',
                                    help="Size of the DRAM")
ps.add_option('--cpu_clk',          type="string", default='1GHz',
                                    help="Speed of all CPUs")

# Instruction Buffer OPTIONS
ps.add_option('--IB_depth',  type="int", default=128,
                                    help="Instruction Buffer depth")
# Elementwise Unit OPTIONS
ps.add_option('--parallel_ew',  type="int", default=4,
                                    help="Number of parallel Elementwise Units")

# MATRIX DISPATCHER OPTIONS
ps.add_option('--MQ_depth',  type="int", default=64,
                                    help="Matrix Dispatcher Memory Queue depth")
ps.add_option('--AQ_depth',  type="int", default=64,
                                    help="Matrix Dispatcher Arithmetic Queue depth")

# MATRIX LANE OPTIONS
ps.add_option('--matrix_engine_clk',  type="string", default="1GHz",
                                    help="Speed of all matrix lane")
ps.add_option('--xbuffer_ports_num',  type="int", default=8,
                                    help="XBuffer Ports Number")
ps.add_option('--ybuffer_ports_num',  type="int", default=8,
                                    help="YBuffer Ports Number")
ps.add_option('--cu_row_size',  type="int", default=8,
                                    help="Compute Unit Array row size, indicate number of row")
ps.add_option('--cu_column_size',  type="int", default=8,
                                    help="Compute Unit Array column size")
ps.add_option('--WIDEN',  type="int", default=4,
                                    help="Data Widen Coeffience")
ps.add_option('--scale_factor',  type="float", default=0.0075,
                                    help="Data scale factor") #default = 0.0075/1
ps.add_option('--enable_activate', type="int", default=True,
                                    help="Enable the activation")
ps.add_option('--activate_mode', type="int", default=0,
                                    help="Activation mode")
ps.add_option('--data_width',  type="int", default=8,
                                    help="XYBuffer data width")
ps.add_option('--buffer_depth',  type="int", default=64,
                                    help="X/YBuffer depth")
ps.add_option('--x_num_port',  type="int", default=8,
                                    help="XBuffer Ports Number")
ps.add_option('--y_num_port',  type="int", default=8,
                                    help="YBuffer Ports Number")
ps.add_option('--z_buffer_depth',  type="int", default=64,
                                    help="ZBuffer depth")
ps.add_option('--z_num_port',  type="int", default=8,
                                    help="ZBuffer Ports Number")

# MATRIX REGISTER
ps.add_option('--tileReg_num',  type="int", default=16,
                                    help="Tile Register Number")
ps.add_option('--accReg_num',  type="int", default=16,
                                    help="Accumulator Register Number")
# ps.add_option('--physicReg_num',  type="int", default=32,
#                                     help="Matrix physical register number")
ps.add_option('--regWidth',  type="int", default=256,
                                    help="Matrix register width")
ps.add_option('--bank_num',  type="int", default=4,
                                    help="Matrix register bank number")
ps.add_option('--bank_depth',  type="int", default=8,
                                    help="Matrix register bank depth")

# MATRIX MMU
ps.add_option('--num_matrix_mem_ports',  type="int", default=1,
                                    help="Matrix memory ports number")
ps.add_option('--w_channel',  type="int", default=0,
                                    help="Matrix write channel")
ps.add_option('--r_channel',  type="int", default=1,
                                    help="Matrix read channel")
ps.add_option('--pending_depth',  type="int", default=64,
                                    help="Matrix MMU pending depth")

# MATRIX RENAME
ps.add_option('--numPhysicalRegs',  type="int", default=32,
                                    help="Matrix physical register number")
ps.add_option('--numLogicalRegs',  type="int", default=8,
                                    help="Matrix logical register number")
# MATRIX ROB
ps.add_option('--ROB_depth',  type="int", default=64,
                                    help="Matrix ROB depth")
ps.add_option('--MDU_depth',  type="int", default=64,
                                    help="Matrix MDU depth")

# MATRIX ENGINE TOP
ps.add_option('--lane_num',  type="int", default=4,
                                    help="Matrix Lane Number")
ps.add_option('--connect_to_l1_d',  type="int", default=False,
                                    help="Connect Matrix Port to L1D")
ps.add_option('--connect_to_l1_m',  type="int", default=True,
                                    help="Connect Matrix Port to L1V")
ps.add_option('--connect_to_l2',    type="int", default=False,
                                    help="Connect Matrix Port to L2")
ps.add_option('--connect_to_dram',  type="int", default=False,
                                    help="Connect Matrix Port to Dram")

# SPM
ps.add_option('--spm_size',         type="string", default='128MB',
                                    help="Size of the SPM")
ps.add_option('--spm_type',         type="int", default=1,
                                    help="SPM type: 1=SimpleMemory, 2=GDDR5")
# Latencies SPM
ps.add_option("--spm-r-lat-1", type="string", default="5ns")
ps.add_option("--spm-r-lat-2", type="string", default="5ns")
ps.add_option("--spm-r-lat-3", type="string", default="5ns")
ps.add_option("--spm-rvar-lat-1", type="string", default="0ns")
ps.add_option("--spm-rvar-lat-2", type="string", default="0ns")
ps.add_option("--spm-rvar-lat-3", type="string", default="0ns")

ps.add_option("--spm-w-lat-1", type="string", default="10ns")
ps.add_option("--spm-w-lat-2", type="string", default="10ns")
ps.add_option("--spm-w-lat-3", type="string", default="10ns")
ps.add_option("--spm-wvar-lat-1", type="string", default="0ns")
ps.add_option("--spm-wvar-lat-2", type="string", default="0ns")
ps.add_option("--spm-wvar-lat-3", type="string", default="0ns")

(options, args) = ps.parse_args()

###############################################################################
# Memory hierarchy configuration
# Here you can select where to connect the vector memmory port, it can be;
# to main memory
# to l2 cache
# to l1 data cache (share with the core)
# to its own vector cache
###############################################################################

connect_to_dram   = options.connect_to_dram
connect_to_l2     = options.connect_to_l2
connect_to_l1d    = options.connect_to_l1_d
connect_to_l1V    = options.connect_to_l1_v

# multiport       = 1
# vector_rf_ports = (((options.num_clusters*5)+3) if multiport else 1)

###############################################################################
# Setup System
###############################################################################

# create the system we are going to simulate
system = System(
    cache_line_size = options.cache_line_size,
    clk_domain = SrcClockDomain(
        clock = options.cpu_clk,
        voltage_domain = VoltageDomain()
    ),
    mem_mode = 'timing',
    mem_ranges = [AddrRange(options.mem_size)]
)

###############################################################################
# CPU CONFIG
###############################################################################
#system.cpu = MinorCPU(mem_unit_channels = mem_unit_channels)
system.cpu = RiscvO3CPU()

###############################################################################
# Create Icache, Dcache, and L2 cache
###############################################################################
system.cpu.icache = Cache(
    size = options.l1i_size,
    assoc = 4,
    tag_latency = 4,
    data_latency = 4,
    response_latency = 4,
    mshrs = 4,
    tgts_per_mshr = 20
)
system.cpu.dcache = Cache(
    size = options.l1d_size,
    assoc = 4,
    tag_latency = 4,
    data_latency = 4,
    response_latency = 4,
    mshrs = 4,
    tgts_per_mshr = 20
)
system.l2cache = Cache(
    size = options.l2_size,
    assoc = 8,
    tag_latency = 6,
    data_latency = 6,
    response_latency = 6,
    mshrs = 20,
    tgts_per_mshr = 12
)

def createMatrixCache(cpu):
    system.MatrixCache = Cache(
        size = "32kB",
        assoc = 4,
        tag_latency = 2,
        data_latency = 2,
        response_latency = 2,
        mshrs = 4,
        tgts_per_mshr = 20
    )

def createbuffer(cpu):
    system.MatrixCache = Cache(
        size = "2kB",
        assoc = 1,
        tag_latency = 0,
        data_latency = 0,
        response_latency = 0,
        mshrs = 4,
        tgts_per_mshr = 20
    )

if( (not connect_to_l1d) and (not connect_to_l2) and (not connect_to_dram)):
    createMatrixCache(system.cpu)

# when matrix engine memory port directly connect to the L2 Cache or DRAM, there should be a small buffer here.
if( (connect_to_l2) or (connect_to_dram)):
    createbuffer(system.cpu)

###############################################################################
# Create buses
###############################################################################
# L2Bus Setting
system.l2bus = L2XBar()

# Create a memory bus
system.membus = SystemXBar()

# # Create a spm bus
# system.spmbus = SpmXBar()

###############################################################################
# Matrix Engine CONFIG
###############################################################################
system.cpu.matrix_interface = MatrixEngineInterface(
    matrix_engine = MatrixEngine(
        matrix_rename = MatrixRename(
            numPhysicalRegs = options.numPhysicalRegs,
            numLogicalRegs = options.numLogicalRegs
        ),
        ew_unit = ElementwiseUnit(
            parallel_ewu = options.lane_num,
            clk_domain = SrcClockDomain(
                clock = options.matrix_engine_clk,
                voltage_domain = VoltageDomain()
            )
        ),
        matrix_rob = ReorderBuffer(
            clk_domain = SrcClockDomain(
                clock = options.matrix_engine_clk,
                voltage_domain = VoltageDomain()  
            ),
            ROB_depth = options.ROB_depth,
            MDU_depth = options.MDU_depth
        ),
        lane_num = options.lane_num,
        matrix_dispatcher = MatrixDispatcher(
            clk_domain = SrcClockDomain(
                clock = options.matrix_engine_clk,
                voltage_domain = VoltageDomain()  
            ),
            MQ_depth = options.MQ_depth,
            AQ_depth = options.AQ_depth
        ),
        matrix_reg= MatrixRF(
            tileReg_num = options.tileReg_num,
            accReg_num = options.accReg_num,
            regWidth = options.regWidth,
            bank_num = options.bank_num,
            bank_depth = options.bank_depth
        ),
        matrix_mmu = MatrixMMU(
            num_matrix_mem_ports = options.num_matrix_mem_ports,
            w_channel = options.w_channel,
            r_channel = options.r_channel,
            pending_depth = options.pending_depth
            # o3cpu = RiscvO3CPU()
        ),
        matrix_lanes = [MatrixLane(
            clk_domain = SrcClockDomain(
                clock = options.matrix_engine_clk,
                voltage_domain = VoltageDomain()  
            ),
            XBuffer = XYBuffer(
                buffer_depth = options.buffer_depth,
                data_width = options.data_width,
                XY = True,
                num_port = options.x_num_port
            ),
            YBuffer = XYBuffer(
                buffer_depth = options.buffer_depth,
                data_width = options.data_width,
                XY = False,
                num_port = options.y_num_port
            ),
            zbuffer = ZBuffer(
                buffer_depth = options.z_buffer_depth,
                data_width = options.data_width,
                num_port = options.z_num_port
            ),
            xbuffer_ports_num = options.xbuffer_ports_num,
            ybuffer_ports_num = options.ybuffer_ports_num,
            cu_row_size = options.cu_row_size,
            cu_column_size = options.cu_column_size,
            WIDEN = options.WIDEN,
            compute_units = [ComputeUnit(
                coordinatex = cu_id % options.cu_column_size,
                coordinatey = cu_id // options.cu_column_size,
                en_activate = options.enable_activate,
                quantization_layer = Quantization(
                    scale_factor = options.scale_factor
                ),
                activation_layer = Activation(
                    mode = options.activate_mode
                )
            )for cu_id in range(0, options.cu_row_size*options.cu_column_size)]
        )for lane_id in range(0, options.lane_num)],
        matrix_spm = MatrixSPM(
            range = m5.objects.AddrRange(start = system.mem_ranges[0].end, size = options.spm_size),
            sram_banks = 4,
            entry_width = 8,
            rw_ports_per_bank = 1,  # only read ports
            r_ports_per_bank = 0,    # only read ports
            w_ports_per_bank = 0,    # only write ports
            entry_depthA = 1024,
            entry_depthB = 1024,
            entry_depthC = 2048,
            dmaDevice = MatrixDmaDevice(
                dma_buffer_width = 32,
                dma_buffer_depth = 32,
                dma_transpose_buffer_num = 4,
                dma_transpose_buffer_width = 32,
                dma_transpose_buffer_depth = 8
            ),
        )
    ),
    inst_buffer = InstructionBuffer(
        IB_depth = options.IB_depth,
        clk_domain = SrcClockDomain(
            clock = options.matrix_engine_clk,
            voltage_domain = VoltageDomain()
        )
    )
)

###############################################################################
# Create MemChecker
###############################################################################
system.memchecker_membus = MemChecker()
system.memcheckermon_membus = MemCheckerMonitor(memchecker=system.memchecker_membus)
system.memchecker_l2cache = MemChecker()
system.memcheckermon_l2cache = MemCheckerMonitor(memchecker=system.memchecker_l2cache)

###############################################################################
# Connect
###############################################################################
system.cpu.icache_port = system.cpu.icache.cpu_side
system.cpu.dcache_port = system.cpu.dcache.cpu_side

system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports
system.cpu.matrix_interface.matrix_engine.matrix_spm.dmaDevice.dmaPort = system.l2bus.cpu_side_ports

system.memcheckermon_l2cache.cpu_side_port = system.l2bus.mem_side_ports
system.memcheckermon_l2cache.mem_side_port = system.l2cache.cpu_side
# system.l2cache.cpu_side = system.l2bus.mem_side_ports

system.l2cache.mem_side = system.memcheckermon_membus.cpu_side_port
system.memcheckermon_membus.mem_side_port = system.membus.cpu_side_ports
# system.l2cache.mem_side = system.membus.cpu_side_ports

# Connect the system up to the membus
system.system_port = system.membus.cpu_side_ports

#create interrupt controller
system.cpu.createInterruptController()

# Create a DDR3 memory controller
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8(device_size = options.mem_size)
# system.mem_ctrl.dram = DDR5_4400_4x8(device_size = options.mem_size)
#system.mem_ctrl = HBM_1000_4H_x64(device_size = options.mem_size)
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# # Connecting spm to spm bus
# system.spm.port = system.spmbus.mem_side_ports
# Connecting matrix engine to spm bus
# system.cpu.matrix_interface.matrix_engine.matrix_mmu.matrix_mem_ports = system.spmbus.cpu_side_ports

###############################################################################
# Create Workload
###############################################################################

filtered = []
#output = []

if not options or not options.cmd:
    print("No --cmd='<workload> [args...]' passed in")
    sys.exit(1)
else:
    split = options.cmd.split(' ')
    for s in options.cmd.split(' '):
      if len(s):
        filtered = filtered + [s]


    # process.executable = filtered[0]
    # process.cmd = filtered
#    process.output = output[0]
numThreads = options.numThreads
RiscvO3CPU().numThreads = numThreads
if numThreads > 1:
    system.multi_thread = True

binary = options.cmd
system.workload = SEWorkload.init_compatible(binary)
process = Process()
process.cmd = [binary]
system.cpu.workload = process
system.cpu.createThreads()

###############################################################################
# Run Simulation
###############################################################################

# set up the root SimObject and start the simulation
root = Root(full_system = False, system = system)
# instantiate all of the objects we've created above
m5.instantiate()

print("Beginning simulation!")
exit_event = m5.simulate()
print('Exiting @ tick %i because %s' % (m5.curTick(), exit_event.getCause()))
print("gem5 finished %s" % datetime.datetime.now().strftime("%b %e %Y %X"))
