#pragma once

#include <cstddef>
#include "../memory/types.hh"

/**
 * cost function class
 * will be replaced by kernels
 */
class CostFunction
{

public:

    virtual ~CostFunction() = default;

    dbl_t* in_get() const;
    const dbl_t* in_hat_get() const;
    dbl_t* out_grad_get() const;
    void in_set(dbl_t* in);
    void in_hat_set(const dbl_t* in_hat);
    void out_grad_set(dbl_t* out_grad);
    void rows_set(std::size_t rows);
    void cols_set(std::size_t cols);


    virtual dbl_t cost() = 0;
    virtual void cost_grad() = 0;

protected:
    dbl_t* in_ = nullptr;
    const dbl_t* in_hat_ = nullptr;
    dbl_t* out_grad_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
};

/**
 * Implement the QuadraticCost or MSE (Mean Squared Error)
 */
class QuadraticCost : public CostFunction
{
public:

    virtual dbl_t cost() override;
    virtual void cost_grad() override;
};

#include "cost-function.hxx"
