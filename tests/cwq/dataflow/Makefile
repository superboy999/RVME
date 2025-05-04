compile:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -funroll-loops -march=rv64gc_zfh_xtheadmatrix -g 128x2048x512.c -o 128x2048x512.elf -O3 -lm -static
exe:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -funroll-loops -march=rv64gc_zfh_xtheadmatrix -g 128x2048x512.c -o 128x2048x512 -O3 -lm -static
dump:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-objdump -S 128x2048x512.elf > 128x2048x512.dump