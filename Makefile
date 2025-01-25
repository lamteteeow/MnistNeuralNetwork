.PHONY: all clean read_dataset_images read_dataset_labels neural_network

ROOT_PATH := .
SRC_PATH   := $(ROOT_PATH)/src
BUILD_PATH := $(ROOT_PATH)/build
BIN_PATH   := $(ROOT_PATH)/bin

INC_PATH  := $(SRC_PATH)/include
INC_DIRS  := $(sort $(shell find $(INC_PATH) -type d))
INC_FLAGS := $(addprefix -iquote ,$(INC_DIRS))

CC := g++
CFLAGS := -Wall -pedantic -Werror -std=c++20 -fopenmp

LDFLAGS := 

all: read_dataset_images read_dataset_labels neural_network

clean:
	rm -rf $(BUILD_PATH) $(BIN_PATH)

read_dataset_images: $(BIN_PATH)/read_dataset_images
read_dataset_labels: $(BIN_PATH)/read_dataset_labels
neural_network: $(BIN_PATH)/neural_network

$(BIN_PATH)/read_dataset_images: $(BUILD_PATH)/read_dataset_images.o
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_PATH)/read_dataset_labels: $(BUILD_PATH)/read_dataset_labels.o
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_PATH)/neural_network: $(BUILD_PATH)/neural_network.o
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.cpp
	mkdir -p $(dir $@)
	$(CC) $(INC_FLAGS) $(CFLAGS) -c $< -o $@
