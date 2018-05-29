#include "mnist.hh"
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include "../memory/alloc.hh"
#include "../memory/copy.hh"

namespace mnist
{

    namespace
    {
        static constexpr std::size_t NIMGS = 70000;
        static constexpr std::size_t IMG_SIZE = 784;
    }
    

    void load(const std::string& path, dbl_t** x, dbl_t** y)
    {

        FILE* f = fopen(path.c_str(), "rb");
        if (!f)
            throw std::runtime_error("mnist: can't open data file");

        *x = tensor_alloc(NIMGS * IMG_SIZE);
        *y = tensor_alloc(NIMGS * 10);

        for (std::size_t i = 0; i < NIMGS; ++i)
        {
            dbl_t* x_row = *x + i * IMG_SIZE;
            dbl_t* y_row = *y + i * 10;

            unsigned char pixs[IMG_SIZE];
            char digit;
            fread(pixs, 1, IMG_SIZE, f);
            fread(&digit, 1, 1, f);

            for (std::size_t i = 0; i < IMG_SIZE; ++i)
                x_row[i] = pixs[i] / 255.0;
            digit_to_vector(digit, y_row);
        }

        fclose(f);
    }

    void digit_to_vector(std::size_t digit, dbl_t* out)
    {
        assert(digit < 10);
        tensor_fill(out, out + 10, 0);
        out[digit] = 1;
    }

    std::size_t vector_to_digit(const dbl_t* v)
    {
        std::size_t res = 0;
        for (std::size_t i = 1; i < 10; ++i)
            if (v[i] > v[res])
                res = i;
        return res;
    }

    bool output_test(const dbl_t* a , const dbl_t* b)
    {
        return vector_to_digit(a) == vector_to_digit(b);
    }

}
