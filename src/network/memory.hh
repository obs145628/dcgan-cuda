#pragma once

#include <cstddef>
#include "../config/types.hh"


dbl_t* tensor_alloc(std::size_t size);
void tensor_free(dbl_t* ptr);
void tensor_fill(dbl_t* begin, dbl_t* end, dbl_t val); 


#include "memory.hxx"
