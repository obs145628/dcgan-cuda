#include "layer.hh"


void Layer::input_set(const dbl_t* input)
{
    input_ = input;
}

void Layer::output_set(dbl_t* output)
{
    output_ = output;
}
