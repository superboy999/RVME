TARGET = layernorm
compile:
	/cluster/home/guoym/xuantie-gnu-toolchain/opt/riscv/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g $(TARGET).c -o $(TARGET).elf -O2 -lm -static
exe:
	/cluster/home/guoym/xuantie-gnu-toolchain/opt/riscv/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g $(TARGET).c -o $(TARGET) -O2 -lm -static
dump:
	/cluster/home/guoym/xuantie-gnu-toolchain/opt/riscv/bin/riscv64-unknown-elf-objdump -D $(TARGET).elf > $(TARGET).dump
	