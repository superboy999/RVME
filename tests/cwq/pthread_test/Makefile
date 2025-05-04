compile:
	/cluster/home/chenwq/research/rvm/riscv-matrix-extension-spec/demos/toolchain/gcc/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gc_zfh_xtheadmatrix -pthread -g pthread_test.c -o pthread_test.elf -O2 -lm -static -pthread
exe:
	/cluster/home/chenwq/research/rvm/riscv-matrix-extension-spec/demos/toolchain/gcc/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gc_zfh_xtheadmatrix -g pthread_test.c -o pthread_test -O2 -lm -static -pthread
dump:
	/cluster/home/chenwq/research/rvm/riscv-matrix-extension-spec/demos/toolchain/gcc/bin/riscv64-unknown-linux-gnu-objdump -S pthread_test.elf > pthread_test.dump
texe:
	/cluster/home/chenwq/research/rvm/riscv-matrix-extension-spec/demos/toolchain/gcc/bin/riscv64-unknown-linux-gnu-g++ -march=rv64gc_zfh_xtheadmatrix -g threads.cpp -o threads -O2 -lm -static -pthread