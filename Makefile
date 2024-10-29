# Compiler and flags
CXX := gcc
CXXFLAGS := -Wall -std=c++17

# Directories
SRC_DIR := ./src
LIB_DIR := ./lib

# Target name
TARGET_NAME := spinw

# Source files and output library
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(LIB_DIR)/%.o, $(SRC_FILES))
LIBRARY := $(LIB_DIR)/$(TARGET_NAME).a

# Default target
all: $(LIBRARY)

# Create static library from object files
$(LIBRARY): $(OBJ_FILES)
	@mkdir -p $(LIB_DIR)
	ar rcs $(LIBRARY) $(OBJ_FILES)

# Compile each .cpp file to .o
$(LIB_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -rf $(LIB_DIR)/*.o $(LIBRARY)

# Rebuild everything
rebuild: clean all

# PHONY targets
.PHONY: all clean rebuild
