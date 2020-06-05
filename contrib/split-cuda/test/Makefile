# NOTE: Update the following variables for your system
CC=gcc
CXX=g++
NVCC=/usr/local/cuda/bin/nvcc
LD=${CXX}
CUDA_INCLUDE_PATH=/usr/local/cuda/include/


# Wrapper library against which the target application will be linked.
WRAPPER_LIB=cuda_wrappers

INCLUDE_FLAGS= -I/home/twinkle/mpich-install/include
WARNING_FLAGS=-Wall -Wno-deprecated-declarations -Werror
NVCC_FLAGS=-ccbin=/usr/bin/g++-7 -Xcompiler '-g3 -O0 $(WARNING_FLAGS)'

FILES=simple_addition malloc_across_kernel read_device_var read_uvm_var \
      rw_device_var rw_dev_array texture_test uvm_var uvm_args \
      uvm_output_args uvm_var_managed texture_object_test 
# cudaCreateChannelDesc_test cuda_plus_mpi

OBJS=$(addsuffix .o, ${FILES})

TESTS_DUMMY=$(addsuffix .dummy.exe, ${FILES})
TESTS=$(addsuffix .exe, ${FILES})
LDFLAGS_DUMMY=-L.. -lcuda_wrappers

default: ${TESTS} ${TESTS_DUMMY}

%.o: %.c
	${NVCC} ${INCLUDE_FLAGS} ${NVCC_FLAGS} -c $< -o $@

%.o: %.cu
	${NVCC} ${INCLUDE_FLAGS} ${NVCC_FLAGS} -c $< -o $@

%.exe: %.cu
	$(NVCC) ${INCLUDE_FLAGS} ${NVCC_FLAGS} --cudart shared -o $@ $<

%.dummy.exe: %.o
	$(LD) -o $@ $< ${LDFLAGS_DUMMY}

tidy:
	rm -f ckpt_*.dmtcp dmtcp_restart_script* \
	dmtcp-shared-memory.* dmtcp-test-typescript.tmp core*
	rm -rf ckpt_*

clean: tidy
	rm -f $(TESTS) ${OBJS} ${TESTS_DUMMY} *.pyc *.so
