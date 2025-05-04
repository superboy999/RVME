compile:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g test.c -o test.elf -O2 -lm -static
exe:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g test.c -o test -O2 -lm -static
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g 128.c -o 128 -O2 -lm -static
dump:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-objdump -S test.elf > test.dump