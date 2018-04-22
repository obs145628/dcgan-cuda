#pragma once

#include <cstddef>
#include "types.hh"


dbl_t* tensor_alloc(std::size_t size);
void tensor_free(dbl_t* ptr);

