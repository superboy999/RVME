compile:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g dispatch_density.c -o dispatch_density.elf -O2 -lm -static
exe:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g dispatch_density.c -o dispatch_density -O2 -lm -static
dump:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-objdump -S dispatch_density.elf > dispatch_density.dump