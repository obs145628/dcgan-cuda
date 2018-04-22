#pragma once

#include "../memory/types.hh"
#include <string>

namespace mnist
{

    /**
     * Read mnist dataset, and allocate space to store it
     * @param path - file path of the mnist file
     * @param x - will contain a matrix (70000 * 784)
     * @param y - will contain a matrix (70000 * 10)
     */
    void load(const std::string& path, dbl_t** x, dbl_t** y);
    
    void digit_to_vector(std::size_t digit, dbl_t* out);
    
    std::size_t vector_to_digit(const dbl_t* v);

    /**
     * Check if the two vectors correspond to the same number
     */
    bool output_test(const dbl_t* a , const dbl_t* b);

}
