#include "layer.hh"

inline const dbl_t* Layer::input() const
{
    return input_;
}

inline dbl_t* Layer::output() const
{
    return output_;
}
