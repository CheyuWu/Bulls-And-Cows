BUILD_DIR := build
TARGET := bull_and_cows
EXECUTABLE := $(BUILD_DIR)/$(TARGET)

.PHONY: all build run clean rebuild

all: build run

$(EXECUTABLE):
	@echo "Generating build system..."
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	@cmake --build $(BUILD_DIR) --config Release

run:
	@$(EXECUTABLE)

clean:
	@rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory."

build: clean $(EXECUTABLE)
