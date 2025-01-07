# Compiler (wrapper) and flags
CC = mpicc
CFLAGS = -Wall -Wextra -I./include

# Source directory
SRC_DIR = ./src

# Where to create the executable
OUT_DIR = .

# Executable name
TARGET = bin

# All source files
SOURCES = $(wildcard $(SRC_DIR)/*.c)

# Generate object file names
OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(TARGET)

# Link target and executable
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(OUT_DIR)/$@

# Compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)
	
# Phony targets
.PHONY: all clean

# Optional: rebuild
rebuild: clean all