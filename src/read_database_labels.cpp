#include "tensor.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
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
 * @brief Entry point for the program {read_dataset_labels.cpp}
 *
 * @param {argc} cmd-line argument count.
 * @param {argv} cmd-line argument values.
 *
 * @return The status code signifies the program return status.
 *
 * */
int main(int argc, const char * argv[]) {

    if ( argc != 2 ) {

        std::cerr 
            << "Usage:" 
            << SPACE
            << "./" << argv[0] 
            << SPACE
            << "<label-file>" << std::endl;

        return 1;

    }

    std::string const label_file_name = argv[1];

    return 0;

}

