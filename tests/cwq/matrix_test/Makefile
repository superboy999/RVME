compile:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g matrix_test.c -o matrix_test.elf -O2 -lm -static
exe:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-gcc -march=rv64gc_zfh_xtheadmatrix -g matrix_test.c -o matrix_test -O2 -lm -static
dump:
	/cluster/home/chenwq/research/rvm/gcc/bin/riscv64-unknown-elf-objdump -S matrix_test.elf > matrix_test.dump