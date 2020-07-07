# NOTE: Update the following variables for your system
CC=/usr/bin/gcc-8
CXX=/usr/bin/g++-8
NVCC=/usr/local/cuda/bin/nvcc
LD=${CXX}
RTLD_PATH=/lib64/ld-linux-x86-64.so.2
CUDA_INCLUDE_PATH=/usr/local/cuda/include/

# The name will be the same as the current directory name.
NAME=${shell basename $$PWD}

# By default, your resulting plugin library will have this name.
LIBNAME=libdmtcp_${NAME}

# Wrapper library against which the target application will be linked.
WRAPPER_LIB=cuda_wrappers

# Modify if your DMTCP_ROOT is located elsewhere.
ifndef DMTCP_ROOT
  DMTCP_ROOT=../..
endif
DMTCP_INCLUDE=${DMTCP_ROOT}/include
JALIB_INCLUDE=${DMTCP_ROOT}/jalib

# Flags for compile and link configurations

NVCC_FLAGS=-Xlinker -Ttext-segment -Xlinker 0x800000 --cudart shared
NVCC_OPTFLAGS=-Xcompiler -g3 -O0

SPLIT_CUDA_INCLUDE_FLAGS=-I${DMTCP_ROOT}/contrib/split-cuda/common \
       -I${DMTCP_ROOT}/contrib/split-cuda/utils \
			 -I${DMTCP_ROOT}/contrib/split-cuda/uh \
       -I${DMTCP_ROOT}/contrib/split-cuda/lh \
			 -I${DMTCP_ROOT}/contrib/split-cuda/log_and_replay \
       -I${DMTCP_ROOT}/contrib/split-cuda/autogen

INCLUDE_FLAGS=-I. -I${CUDA_INCLUDE_PATH} -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} \
              ${SPLIT_CUDA_INCLUDE_FLAGS}
WARNING_FLAGS=-Wall -Wno-deprecated-declarations -Werror

override CFLAGS += -g3 -O0 -fPIC ${INCLUDE_FLAGS} -c -std=gnu11 \
                ${WARNING_FLAGS} -fno-stack-protector
override CXXFLAGS += -g3 -O0 -fPIC ${INCLUDE_FLAGS} -c -std=c++11 \
                  ${WARNING_FLAGS} -fno-stack-protector

# variables related to kernel loader
KERNEL_LOADER_OBJS=lh/kernel-loader.o utils/procmapsutils.o lh/custom-loader.o \
		   lh/mmap-wrapper.o lh/sbrk-wrapper.o lh/cuda-lh-if.o utils/utils.o \
		   utils/trampoline_setup.o lh/mem-restore.o lh/replay_cuda_calls.o \
       uh/switch_context.o

KERNEL_LOADER_CFLAGS=-DSTANDALONE
KERNEL_LOADER_BIN=kernel-loader.exe

# variables related to small cuda application (target.c)
TARGET_OBJS=target.o
TARGET_BIN=target.dummy.exe

# variables related to autogen code
AUTOGEN_OBJS=autogen/cuda_autogen_wrappers.o


# variables related to stub library which should be linked with user's cuda-
# application; This library has dummy wrappers for cuda APIs
WRAPPER_LIB_OBJS=cuda_stub_lib/cuda_stub.o

# variables related to dmtcp's cuda-plugin (upper-half)
UTILITY_OBJS=utils/procmapsutils.o utils/utils.o \
        utils/trampoline_setup.o
PLUGIN_LIB_OBJS=${AUTOGEN_OBJS} ${UTILITY_OBJS}  \
                uh/upper-half-wrappers.o uh/cuda-plugin.o \
                log_and_replay/log_cuda_api.o uh/switch_context.o

#variables for make check
DMTCP_LAUNCH=${DMTCP_ROOT}/bin/dmtcp_launch
DMTCP_LAUNCH_FLAGS=--new-coordinator --cuda --kernel-loader \
		   $$PWD/${KERNEL_LOADER_BIN} --target-ld ${RTLD_PATH} \
		   --with-plugin $$PWD/${LIBNAME}.so


default: ${AUTOGEN_OBJS} ${WRAPPER_LIB_OBJS} ${LIBNAME}.so ${KERNEL_LOADER_BIN}\
         lib${WRAPPER_LIB}.so ${TARGET_BIN}

autogen: autogen/cuda_autogen_wrappers.o
autogen/cuda_autogen_wrappers.o: autogen/cuda_autogen_wrappers.cpp
	${CXX} ${CXXFLAGS} -Wno-unused-variable -c $< -o $@
autogen/cuda_autogen_wrappers.cpp: autogen/generate-split-proc-wrappers.py \
                           autogen/cuda_wrapper_signatures.txt
	python autogen/generate-split-proc-wrappers.py \
  autogen/cuda_wrapper_signatures.txt --default \
  > autogen/cuda_autogen_wrappers.cpp ;

cuda_stub_lib/cuda-stub.o: cuda_stub_lib/cuda_stub.cpp
	${CXX} ${CXXFLAGS} -c $< -o $@

${LIBNAME}.so: ${PLUGIN_LIB_OBJS}
	${CXX} -shared -fPIC -g3 -O0 -o $@ $^

lib${WRAPPER_LIB}.so: ${WRAPPER_LIB_OBJS}
	${CC} -shared -fPIC -g3 -O0 -o $@ $^

disableASLR:
	@- [ `cat /proc/sys/kernel/randomize_va_space` = 0 ] \
	|| sudo sh -c 'echo 0 > /proc/sys/kernel/randomize_va_space'

enableASLR:
	@- [ `cat /proc/sys/kernel/randomize_va_space` != 2 ] \
	&& sudo sh -c 'echo 2 > /proc/sys/kernel/randomize_va_space'

check: default
	${DMTCP_LAUNCH} ${DMTCP_LAUNCH_FLAGS} $$PWD/test/${TARGET_BIN}

.c.o:
	${CC} ${CFLAGS} $< -o $@

.cpp.o:
	${CXX} ${CXXFLAGS} $< -o $@

${TARGET_OBJS}: test/target.c
	${NVCC} ${INCLUDE_FLAGS} ${NVCC_OPTFLAGS} -c $< -o test/$@

${TARGET_BIN}: ${TARGET_OBJS}
	${LD} test/$< -o test/$@ -L. -l${WRAPPER_LIB};
	${NVCC} -g test/$< -lcudart -o test/target.exe


# Apparently, Nvidia libraries don't like -pie; so, we are forced
# to link the kernel loader (which is really just emulating the lower
# half) to a fixed address (0x800000)
${KERNEL_LOADER_BIN}: ${KERNEL_LOADER_OBJS}
	#${NVCC} ${NVCC_FLAGS} -L/home/alienware/Documents/CRAC-monitored/CRAC-early-development/contrib/split-cuda/mylib/ $^ -o $@ -lcuda -ldl -lcublas -lcusparse -lcusolver -lwrapper
	${NVCC} ${NVCC_FLAGS}  $^ -o $@ -lcuda -ldl  -lcusparse -lcusolver -lcublas

vi vim:
	vim ${FILE}.cpp

tags:
	gtags .

dist: clean
	(dir=`basename $$PWD` && cd .. && tar zcvf $$dir.tgz $$dir)
	(dir=`basename $$PWD` && ls -l ../$$dir.tgz)

tidy:
	rm -f ./ckpt.img *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp \
	ckpt_*.dmtcp.temp lhInfo* uhInfo* *.exe

clean: tidy
	rm -f ${KERNEL_LOADER_OBJS} test/${TARGET_OBJS} ${KERNEL_LOADER_BIN} \
	      test/${TARGET_BIN} ${PLUGIN_LIB_OBJS} ${UTILITY_OBJS} \
		  ${LIBNAME}.so GTAGS GRTAGS GPATH autogen/cuda_autogen_wrappers.cpp \
		  autogen/cuda_autogen_wrappers.h autogen/lower_half_cuda_if.h \
      lib${WRAPPER_LIB}.so ${WRAPPER_LIB_OBJS} cuda_stub.*

.PHONY: dist vi vim clean gdb tags tidy restart run enableASLR disableASLR check
