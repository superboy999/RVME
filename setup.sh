#!/bin/bash

# 检查是否提供了足够的参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <node_name> <number_of_cpus>"
    exit 1
fi

# 从命令行参数中读取CPU数和节点名
cpus=$2
node=$1

# 使用srun命令获取资源并在资源上执行一系列命令
srun -p cpu -c "$cpus" -w "$node" --pty /bin/bash -c "singularity exec ../../../../dockerfile/gem5-valgrind-gdb.sif /bin/bash -c 'cd ../run && /bin/bash'"

