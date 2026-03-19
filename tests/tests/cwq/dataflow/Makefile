compile:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -funroll-loops -march=rv64gc_zfh_xtheadmatrix -g test1.c -o test1.elf -O3 -lm -static
exe:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -funroll-loops -march=rv64gc_zfh_xtheadmatrix -g 256x512x2048.c -o 256x512x2048 -O3 -lm -static
dump:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-objdump -S test1.elf > test1.dump