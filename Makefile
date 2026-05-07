# Orion-Style GPU Scheduler Makefile
# 
# 使用方法:
#   make              - 构建所有目标
#   make clean        - 清理构建产物
#   make test         - 运行测试
#   make install      - 安装到 /usr/local
#
# 环境变量:
#   CUDA_PATH         - CUDA 安装路径 (默认: /usr/local/cuda)
#   CUDNN_PATH        - cuDNN 安装路径 (默认: 使用 CUDA_PATH)
#   DEBUG             - 如果设置，编译调试版本

# 配置
CUDA_PATH ?= /usr/local/cuda
CUDNN_PATH ?= $(CUDA_PATH)

# 编译器
CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc

# 目录
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj

# 源文件
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CU_SRCS))
OBJS := $(CPP_OBJS) $(CU_OBJS)

# 目标库
TARGET_LIB := $(BUILD_DIR)/libgpu_scheduler.so
TEST_BIN := $(BUILD_DIR)/test_scheduler

# 编译标志
CXXFLAGS := -std=c++17 -fPIC -Wall -Wextra
CXXFLAGS += -I$(INC_DIR)
CXXFLAGS += -I$(CUDA_PATH)/include
CXXFLAGS += -I$(CUDNN_PATH)/include

# 链接标志
LDFLAGS := -shared -fPIC
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lcuda
LDFLAGS += -L$(CUDNN_PATH)/lib64 -lcudnn
LDFLAGS += -lcublas -lcublasLt
LDFLAGS += -ldl -lpthread

# NVCC 标志
NVCCFLAGS := -std=c++17 --compiler-options -fPIC
NVCCFLAGS += -I$(INC_DIR)

# 调试/发布模式
ifdef DEBUG
    CXXFLAGS += -g -O0 -DDEBUG
    NVCCFLAGS += -g -G -O0
else
    CXXFLAGS += -O3 -DNDEBUG
    NVCCFLAGS += -O3
endif

# 注意内存使用: 限制并行任务数以避免内存溢出
# 32GB 内存，保守起见使用 2 个并行任务
MAKEFLAGS += -j2

# ============================================================================
# 目标
# ============================================================================

.PHONY: all clean test install help

all: $(TARGET_LIB)

# 创建目录
$(BUILD_DIR) $(OBJ_DIR):
	mkdir -p $@

# 编译 C++ 源文件
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo "CXX $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 编译 CUDA 源文件 (如果有)
$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@echo "NVCC $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 链接共享库
$(TARGET_LIB): $(OBJS) | $(BUILD_DIR)
	@echo "LD $@"
	$(CXX) $(OBJS) $(LDFLAGS) -o $@
	@echo "Built: $@"

# 测试程序
$(TEST_BIN): $(TARGET_LIB) test/test_main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) test/test_main.cpp -L$(BUILD_DIR) -lgpu_scheduler -Wl,-rpath,'$$ORIGIN' -o $@

test: $(TEST_BIN)
	@echo "Running tests..."
	cd $(BUILD_DIR) && ./test_scheduler

# 安装
install: $(TARGET_LIB)
	install -d /usr/local/lib
	install -d /usr/local/include/orion
	install -m 755 $(TARGET_LIB) /usr/local/lib/
	install -m 644 $(INC_DIR)/*.h /usr/local/include/orion/
	ldconfig

# 清理
clean:
	rm -rf $(BUILD_DIR)

# 帮助
help:
	@echo "Orion-Style GPU Scheduler Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build the shared library (default)"
	@echo "  test     - Build and run tests"
	@echo "  install  - Install to /usr/local"
	@echo "  clean    - Remove build artifacts"
	@echo ""
	@echo "Environment Variables:"
	@echo "  CUDA_PATH  - CUDA installation path (default: /usr/local/cuda)"
	@echo "  CUDNN_PATH - cuDNN installation path (default: CUDA_PATH)"
	@echo "  DEBUG      - Build debug version if set"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make                           # Build release version"
	@echo "  make DEBUG=1                   # Build debug version"
	@echo "  make CUDA_PATH=/opt/cuda-11.8  # Use custom CUDA path"
	@echo ""
	@echo "Using the library:"
	@echo "  LD_PRELOAD=$(BUILD_DIR)/libgpu_scheduler.so python your_script.py"

# 显示配置
.PHONY: config
config:
	@echo "Configuration:"
	@echo "  CUDA_PATH  = $(CUDA_PATH)"
	@echo "  CUDNN_PATH = $(CUDNN_PATH)"
	@echo "  CXX        = $(CXX)"
	@echo "  NVCC       = $(NVCC)"
	@echo "  CXXFLAGS   = $(CXXFLAGS)"
	@echo "  LDFLAGS    = $(LDFLAGS)"
	@echo "  Sources    = $(CPP_SRCS) $(CU_SRCS)"
