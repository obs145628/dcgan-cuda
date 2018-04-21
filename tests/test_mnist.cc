#include <iostream>

#include <tocha/tensor.hh>

#include "../src/network/cost-function.hh"
#include "../src/math_cpu/mat.hh"

#include "../src/network/memory.hh"
#include "../src/network/layer.hh"
#include "../src/network/fully-connected-layer.hh"
#include "../src/network/network.hh"
#include "../src/network/cost-function.hh"
#include "../src/network/activation.hh"
#include "../src/datasets/mnist.hh"

#include <algorithm>

int main(int argc, char** argv)
{
    if (argc != 4)
    {
	std::cerr << "Invalid number of arguments\n";
	return 1;
    }

    dbl_t* x_train;
    dbl_t* y_train;

    mnist::load(argv[1], &x_train, &y_train);
    auto weights = tocha::Tensors::load(argv[2]);


    auto l1 = new FullyConnectedLayer(784, 100, new SigmoidActivation);
    auto l2 = new FullyConnectedLayer(100, 10, new SigmoidActivation);

    dbl_t* w1 = reinterpret_cast<dbl_t*>(weights.arr()[0].data);
    std::copy(w1, w1 + 784 * 100, l1->w_get());
    dbl_t* b1 = reinterpret_cast<dbl_t*>(weights.arr()[1].data);
    std::copy(b1, b1 + 100, l1->b_get());
    dbl_t* w2 = reinterpret_cast<dbl_t*>(weights.arr()[2].data);
    std::copy(w2, w2 + 100 * 10, l2->w_get());
    dbl_t* b2 = reinterpret_cast<dbl_t*>(weights.arr()[3].data);
    std::copy(b2, b2 + 10, l2->b_get());

    Network net({l1, l2}, new QuadraticCost, nullptr);

    dbl_t loss = net.compute_loss(20, x_train, y_train);
    
    tocha::Tensors out;
    out.add(tocha::Tensor::f32(20, 10));
    dbl_t* y_hat = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    std::copy(net.output_get(), net.output_get() + 20 * 10, y_hat);

    out.add(tocha::Tensor::f32(1));
    dbl_t* lossp = reinterpret_cast<dbl_t*>(out.arr()[1].data);
    *lossp = loss;
    
    out.save(argv[3]);

    tensor_free(x_train);
    tensor_free(y_train);
}
