#pragma once

#include "cost-function.hh"

inline dbl_t* CostFunction::in_get() const
{
    return in_;
}

inline const dbl_t* CostFunction::in_hat_get() const
{
    return in_hat_;
}

inline dbl_t* CostFunction::out_grad_get() const
{
    return out_grad_;
}

inline void CostFunction::in_set(dbl_t* in)
{
    in_ = in;
}

inline void CostFunction::in_hat_set(const dbl_t* in_hat)
{
    in_hat_ = in_hat;
}

inline void CostFunction::out_grad_set(dbl_t* out_grad)
{
    out_grad_ = out_grad;
}

inline void CostFunction::rows_set(std::size_t rows)
{
    rows_ = rows;
}

inline void CostFunction::cols_set(std::size_t cols)
{
    cols_ = cols;
}
