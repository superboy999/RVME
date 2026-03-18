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

# # VECTOR REGISTER OPTIONS
# ps.add_option('--renamed_regs',     type="int", default=40,
#                                     help="Number of Vector Physical Registers")
# ps.add_option('--VRF_line_size',    type="int", default=8,
#                                     help="Vector Register Slice line size in Bytes (per lane)")
# # VECTOR QUEUES OPTIONS
# ps.add_option('--OoO_queues',       type="int", default=True,
#                                     help="Out-of-Order/In-Order Queues")
# ps.add_option('--mem_queue_size',   type="int", default=32,
#                                     help="Memory Queues")
# ps.add_option('--arith_queue_size', type="int", default=32,
#                                     help="Vector Arithmetic")
# # REORDER BUFFER OPTIONS
# ps.add_option('--rob_size',         type="int", default=64,
#                                     help="Reorder Buffer size")

# # VECTOR EXECUNIT OPTIONS
# ps.add_option('--vector_clk',       type="string", default='1GHz',
#                                     help="Speed of Vector Accelerator")
# ps.add_option('--v_lanes',          type="int", default=8,
#                                     help="Number of Lanes")
# ps.add_option('--max_vl',           type="int", default=16384,
#                                     help="Maximum Vector Lenght in bits")
# ps.add_option('--num_clusters',     type="int", default=1,
#                                     help="Number execution clusters")

# ps.add_option('--connect_to_l1_d',  type="int", default=False,
#                                     help="Connect Vector Port to L1D")
# ps.add_option('--connect_to_l1_v',  type="int", default=False,
#                                     help="Connect Vector Port to L1V")
# ps.add_option('--connect_to_l2',    type="int", default=True,
#                                     help="Connect Vector Port to L2")
# ps.add_option('--connect_to_dram',  type="int", default=False,
#                                     help="Connect Vector Port to Dram")

(options, args) = ps.parse_args()

###############################################################################
# Memory hierarchy configuration
# Here you can select where to connect the vector memmory port, it can be;
# to main memory
# to l2 cache
# to l1 data cache (share with the core)
# to its own vector cache
###############################################################################

# connect_to_dram   = options.connect_to_dram
# connect_to_l2     = options.connect_to_l2
# connect_to_l1d    = options.connect_to_l1_d
# connect_to_l1V    = options.connect_to_l1_v

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
system.cpu.matrix_interface = MatrixEngineInterface(
   matrix_engine = MatrixEngine()
)

###############################################################################
# Create CPU and add simple Icache and Dcache
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
    tag_latency = 12,
    data_latency = 12,
    response_latency = 12,
    mshrs = 20,
    tgts_per_mshr = 12
)

# L2Bus Setting
system.l2bus = L2XBar()

# Create a memory bus
system.membus = SystemXBar()

###############################################################################
# Connect
###############################################################################
system.cpu.icache_port = system.cpu.icache.cpu_side
system.cpu.dcache_port = system.cpu.dcache.cpu_side

system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

system.l2cache.mem_side = system.membus.cpu_side_ports
system.l2cache.cpu_side = system.l2bus.mem_side_ports

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
