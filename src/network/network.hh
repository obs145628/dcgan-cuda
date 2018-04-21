#pragma once

#include "fwd.hh"
#include "../config/types.hh"

#include <functional>

#include <iostream>

class Network
{

public:

    using test_f = std::function<bool(const dbl_t*, const dbl_t*)>;
    
    Network(const layers_t& layers, CostFunction* cost, test_f test_function);
    ~Network();

    layers_t& layers_get();
    const layers_t& layers_get() const;
    dbl_t* output_get() const;

    void clean_tensors();
    std::size_t ninputs_get() const;
    void ninputs_set(std::size_t ninputs);

    /**
     * Apply a feed forward pass through all layers
     * Only take one input at a time
     * @param x (input_size)
     * #return (output_size)
     */
    void forward();
    void forward(std::size_t ninputs, const dbl_t* input);
    dbl_t compute_loss(std::size_t ninputs, const dbl_t* x, const dbl_t* y);

    /**
     * Apply back propagation algorithm to one training example x
     * x (input_size) - input data
     * y (output_size) - expected output data
     * Compute the gradient of the loss function for each parameters of each layer
     */
    //void backpropagation(const Vector& x, const Vector& y);

    /**
     * Compute the cost error fnction for a whole data set of size n
     * @param x (n, input_size)
     * @param y (n, output_size)
     * @return cost error value, with regularisation
     *
     * C(x, y) = sum(x, cost(x)) 
     * + (lambda1 * (1/N) * ||W_1||) 
     * + (lambda2 * (1/2N) * ||W||_2^2)
     */
    //num_t data_cost(const Matrix& x, const Matrix& y);


    /**
     * Evaluate the data set and display the results of the NN
     * @param x (n, input_size)
     * @param y (n, output_size)
     */
    //void evaluate(const Matrix& x, const Matrix& y);


    /**
     * Train and evaluate the network for <epochs> epochs
     * @param x_train (n_train, input_size)
     * @param y_train (n_train, output_size)
     * @param x_test (n_test, input_size)
     * @param y_test (n_test, output_size)
     * @param opti - optimizer
     * @param epochs - number of epochs
     * @param disp_train_cost - display the cost function on the training test
     */
    //void train(Matrix& x_train, Matrix& y_train, Matrix& x_test, Matrix& y_test,
//	       Optimizer& opti, std::size_t epochs,
//	       bool disp_train_cost = false);

    /**
     * Check backpropagation on each layer of the neural network
     * @param x (n, input_size) - matrix of the whole input data
     * @param y (n, output_size) - matrix of the whole output data
     * @return if all layers are correct
     */
    //bool check_backpropagation(const Matrix& x, const Matrix& y);
    
private:
    layers_t layers_;
    CostFunction* cost_;
    test_f test_function_;
    std::size_t ninputs_;

    dbl_t* input_;
    dbl_t* output_;
};

