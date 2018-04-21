#pragma once

#include <cstddef>
#include "../config/types.hh"

class CostFunction
{

public:

    virtual ~CostFunction() = default;

    virtual const char* name() const = 0;

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

    /**
     * Compute the cost error derivative at output layer (dC/da)
     * @param y (output_size) - expected output vector
     * @param y_hat (output_size, 1) - output vector of the network
     * @return (output_size) - vector dC/da
     */
    //virtual Vector cost_prime(const Vector& y, const Vector& y_hat) const = 0;

protected:
    dbl_t* in_ = nullptr;
    const dbl_t* in_hat_ = nullptr;
    dbl_t* out_grad_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
};

class QuadraticCost : public CostFunction
{
public:
    const char* name() const override;

    virtual dbl_t cost() override;
    virtual void cost_grad() override;
};

/*
class CrossEntropyCost : public CostFunction
{
public:
    const char* name() const override;
    num_t cost(const Vector& y, const Vector& y_hat) const override;
    Vector cost_prime(const Vector& y, const Vector& y_hat) const override;
};
*/

#include "cost-function.hxx"
