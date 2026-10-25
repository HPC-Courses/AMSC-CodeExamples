# Usage: make BACKEND=gpu (default), or make BACKEND=cpu
# CPU builds use Thrust's sequential C++ backend; CUDA/Thrust headers are
# still required, but running the CPU executable does not require a GPU.
BACKEND ?= gpu
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= sm_89
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXXFLAGS ?= -O3 -std=c++20
CPPFLAGS += -I$(CUDA_HOME)/include -I$(CUDA_HOME)/include/cccl

ifeq ($(BACKEND),gpu)
COMPILER = $(NVCC)
BACKEND_FLAGS = -x cu --extended-lambda -arch=$(CUDA_ARCH) \
                -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
else ifeq ($(BACKEND),cpu)
COMPILER = $(CXX)
BACKEND_FLAGS = -x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
else
$(error Invalid BACKEND '$(BACKEND)'; choose gpu or cpu)
endif

# Rebuild so switching BACKEND never leaves an executable for the other mode.
.PHONY: all clean FORCE
all: dot_products_with_zip

dot_products_with_zip: dot_products_with_zip.cu Makefile FORCE
	$(COMPILER) $(CPPFLAGS) $(CXXFLAGS) $(BACKEND_FLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)
FORCE:

clean:
	$(RM) dot_products_with_zip
