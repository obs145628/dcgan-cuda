#pragma once

#include <cstddef>
#include "types.hh"

/**
 * Allocates memory dynamically for a tensor
 */
dbl_t* tensor_alloc(std::size_t size);

/**
 * Free dynamically allocated memory of a tensor
 */
void tensor_free(dbl_t* ptr);

