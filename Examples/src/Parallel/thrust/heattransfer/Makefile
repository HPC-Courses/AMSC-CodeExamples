# NVIDIA GeForce RTX 4060 Laptop GPU: compute capability 8.9.
# Override for another GPU, e.g. make CUDA_ARCH=sm_86.
NVCC ?= nvcc
CXX ?= g++
CUDA_ARCH ?= sm_89
CXXSTD ?= c++17

# Override these paths if the dependencies are installed elsewhere.
GETPOT_DIR ?= $(HOME)/Dropbox/Work/Didattica/HPC_Courses/pacs-examples/Examples/include
JSON_DIR ?= /usr/local/include/nlohmann
CUDA_HOME ?= /usr/local/cuda
THRUST_DIR ?= $(CUDA_HOME)/include/cccl

TARGET := main
HOST_TARGET := main_host
BUILD_DIR := .build
OBJECTS := $(BUILD_DIR)/main.o $(BUILD_DIR)/readParameters.o $(BUILD_DIR)/parameters.o
HOST_OBJECTS := $(BUILD_DIR)/main_host.o $(BUILD_DIR)/readParameters.o $(BUILD_DIR)/parameters.o
DEPS := $(sort $(OBJECTS:.o=.d) $(HOST_OBJECTS:.o=.d))

CPPFLAGS ?=
CXXFLAGS ?= -O3 -Wall -Wextra -march=native
NVCCFLAGS ?= -O3
LDFLAGS ?=
LDLIBS ?=
INCLUDES := -I"$(GETPOT_DIR)" -I"$(JSON_DIR)"
CUDA_FLAGS := -std=$(CXXSTD) -arch=$(CUDA_ARCH) --extended-lambda -ccbin $(CXX)

.DEFAULT_GOAL := all
.PHONY: all gpu host clean
all: $(TARGET)
gpu: $(TARGET)
host: $(HOST_TARGET)

$(TARGET): $(OBJECTS) Makefile
	$(NVCC) $(CUDA_FLAGS) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

# The CPP backend stores device_vector data on the host and runs algorithms
# sequentially. Strip CUDA's device annotation only for this g++ compilation;
# the source is shared unchanged, and the executable does not link CUDA.
$(HOST_TARGET): $(HOST_OBJECTS) Makefile
	$(CXX) $(LDFLAGS) -o $@ $(HOST_OBJECTS) $(LDLIBS)

$(BUILD_DIR)/main_host.o: main.cu Makefile | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) -I"$(THRUST_DIR)" -I"$(CUDA_HOME)/include" $(CXXFLAGS) -std=$(CXXSTD) -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -D__device__= -x c++ -MMD -MP -c $< -o $@

$(BUILD_DIR)/main.o: main.cu Makefile | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(INCLUDES) $(NVCCFLAGS) $(CUDA_FLAGS) -MMD -MP -c $< -o $@

# These translation units are ordinary C++, compiled without CUDA.
$(BUILD_DIR)/%.o: %.cpp Makefile | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) -std=$(CXXSTD) -MMD -MP -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	$(RM) $(sort $(OBJECTS) $(HOST_OBJECTS)) $(DEPS) $(TARGET) $(HOST_TARGET)

-include $(DEPS)
