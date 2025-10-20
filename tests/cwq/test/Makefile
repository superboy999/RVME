compile:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g test2.c -o test2.elf -O2 -lm -static
exe:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g test2.c -o test2 -O2 -lm -static
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g test.c -o test -O2 -lm -static
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g test1.c -o test1 -O2 -lm -static
dump:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-objdump -S test2.elf > test2.dump