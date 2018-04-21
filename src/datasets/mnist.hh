#pragma once

#include "../config/types.hh"
#include <string>

namespace mnist
{

    void load(const std::string& path, dbl_t** x, dbl_t** y);
    void digit_to_vector(std::size_t digit, dbl_t* out);
    std::size_t vector_to_digit(const dbl_t* v);
    bool output_test(const dbl_t* a , const dbl_t* b);

}
