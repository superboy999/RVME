compile:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g encoder.c -o encoder.elf -O2 -lm -static
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g test.c -o test.elf -O2 -lm -static
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g tiny.c -o tiny.elf -O2 -lm -static
exe:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g encoder.c -o encoder -O2 -lm -static
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g test.c -o test -O2 -lm -static
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g tiny.c -o tiny -O2 -lm -static
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g tiling_test.c -o tiling_test -O2 -lm -static
dump:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-objdump -S encoder.elf > encoder.dump
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-objdump -S test.elf > test.dump
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-objdump -S tiny.elf > tiny.dump
