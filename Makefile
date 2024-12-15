.PHONY: all clean read_dataset_images

ROOT_PATH := .
SRC_PATH   := $(ROOT_PATH)/src
BUILD_PATH := $(ROOT_PATH)/build
BIN_PATH   := $(ROOT_PATH)/bin

INC_PATH  := $(SRC_PATH)/include
INC_DIRS  := $(sort $(shell find $(INC_PATH) -type d))
INC_FLAGS := $(addprefix -iquote ,$(INC_DIRS))

CC := clang++
CFLAGS := -Wall -pedantic -Werror -std=c++20

LDFLAGS := 

all: read_dataset_images

clean:
	rm -rf $(BUILD_PATH) $(BIN_PATH)

read_dataset_images: $(BIN_PATH)/read_dataset_images

$(BIN_PATH)/read_dataset_images: $(BUILD_PATH)/read_dataset_images.o
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.cpp
	mkdir -p $(dir $@)
	$(CC) $(INC_FLAGS) $(CFLAGS) -c $< -o $@

