#include "tensor.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define SPACE (" ")

/**
 * @author Junzhe Wang
 * @since 16.12.2024
 *
 * @brief Converts a 32-bit unsigned integer from big-endian format into little-endian format
 *
 * @param {big-endian} The 32-bit unsigned integer in big-endian format.
 *
 * @return The 32-bit unsigned integer in little-endian format.
 * */
uint32_t big_endian_to_lit_endian(uint32_t big_endianian) {

    uint32_t little_endian = 0;

    little_endian |= (big_endianian & 0xff'00'00'00) >> 24;
    little_endian |= (big_endianian & 0x00'ff'00'00) >> 8;
    little_endian |= (big_endianian & 0x00'00'ff'00) << 8;
    little_endian |= (big_endianian & 0x00'00'00'ff) << 24;

    return little_endian;

}

/**
 * @author Junzhe Wang
 * @since 16.12.2024
 * 
 * @brief Reads MNIST label data from a binary file and stores it as one-hot encoded tensors.
 *
 * @param {label_file_name} The name of the binary MNIST label file.
 * @param {labels} A vector to store one-hot encoded tensors for each label.
 * 
 * @return The number of labels read from the file.
 */
uint32_t label_rd(std::string const &label_file_name, std::vector<Tensor<double>> &labels)
{

    std::ifstream input( label_file_name , std::ios::binary );
    if ( !input.is_open() ) {

        std::cerr 
            << "Error:"
            << SPACE
            << "Failed to open file [" << label_file_name << "]." 
            << std::endl;

        exit( EXIT_FAILURE );

    }

    uint32_t MAGIC;
    uint32_t ITEM_COUNT;

    input.read( reinterpret_cast<char*>( &MAGIC )     , 4 );
    input.read( reinterpret_cast<char*>( &ITEM_COUNT ), 4 );

    MAGIC = big_endian_to_lit_endian( MAGIC );
    ITEM_COUNT = big_endian_to_lit_endian( ITEM_COUNT );

    if ( MAGIC != 0x0000'0801 ) {

        std::cerr 
            << "Error: Failed to read labels."
            << SPACE
            << "File [" << label_file_name << "] is not in valid format!"
            << std::endl;

        exit( EXIT_FAILURE );

    }

    std::cout << 1 << std::endl;
    std::cout << 10 << std::endl;

    labels.reserve( ITEM_COUNT );
    for (uint32_t i = 0; i < ITEM_COUNT; i++) {

        uint8_t label;
        input.read( reinterpret_cast<char*>( &label ), 1 );

        Tensor<double> one_hot( {10}, 0.0 );
        one_hot({ label }) = 1.0;

        labels.push_back( std::move( one_hot ) );

    }

    input.close();

    return ITEM_COUNT;
}

/**
 * @author Junzhe Wang
 * @since 16.12.2024
 * 
 * @brief Displays a one-hot encoded label tensor in a human-readable format.
 *
 * @param {tensor} The tensor representing the one-hot encoded label.
 *
 * @return None
 */
void display_label_tensor( Tensor<double> const& tensor ) {

    auto const& shape = tensor.shape();
    bool const check = (shape.size() == 1) && (shape[0] == 10);
    if ( !check ) {

        std::cerr << "Error: Tensor does NOT possess the expected 1D shape {10}!";

        exit( EXIT_FAILURE );

    }

    for (uint32_t i = 0; i < shape[0]; i++) std::cout << tensor({ i }) << std::endl;

}

/**
 * @author Junzhe Wang, Lam Tran
 * @since 26.01.2025
 *
 * @brief Entry point for the program {read_dataset_labels.cpp}
 *
 * @param {argc} cmd-line argument count.
 * @param {argv} cmd-line argument values.
 *
 * @return The status code signifies the program return status.
 *
 * */
int main(int argc, const char *argv[])
{
    if (argc != 2 && argc != 3)
    {
        std::cerr
            << "Usage:"
            << SPACE
            << "./" << argv[0]
            << SPACE
            << "<image-file>"
            << SPACE
            << "[<image-index>]" << std::endl;
        return 1;
    }

    std::string const label_file_name = argv[1];
    int label_index = -1;

    if (argc == 3)
    {
        label_index = std::stoi(argv[2]);
    }

    std::vector<Tensor<double>> labels;
    uint32_t const LABEL_COUNT = label_rd(label_file_name, labels);

    if (label_index >= 0)
    {
        if (label_index >= static_cast<int>(LABEL_COUNT))
        {
            std::cerr << "Label index out of range" << std::endl;
            return 1;
        }
        display_label_tensor(labels[label_index]);
    }
    else
    {
        for (uint32_t i = 0; i < LABEL_COUNT; i++)
        {
            display_label_tensor(labels[i]);
        }
    }

    return 0;
}