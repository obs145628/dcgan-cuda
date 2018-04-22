#pragma once

#include "types.hh"

/**
 * Fill a whole tensor with the same value
 */
void tensor_fill(dbl_t* begin, dbl_t* end, dbl_t val);

/**
 * Copy data from memory to tensor
 */
void tensor_write(dbl_t* obegin, dbl_t* oend, const dbl_t* ibegin);

/**
 * Copy data from tensor to memory
 */
void tensor_read(const dbl_t* ibegin, const dbl_t* iend, dbl_t* obegin);

#include "copy.hxx"
