#include <iostream>

#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/graph.hh"
#include "../src/api/activ.hh"
#include "../src/api/layers.hh"
#include "../src/api/cost.hh"
#include "../src/api/sgd-optimizer.hh"

#include "../src/datasets/mnist.hh"
#include "../src/memory/alloc.hh"


int main(int argc, char** argv)
{
    if (argc != 2)
    {
	std::cerr << "Usage: ./nn_mnist <mnist-file>\n";
	return 1;
    }

    dbl_t* x_train;
    dbl_t* y_train;
    mnist::load(argv[1], &x_train, &y_train);

    auto& graph = ops::Graph::instance();
    //graph.debug_set(true);
    auto& builder = ops::OpsBuilder::instance();
    
    auto x = builder.input(ops::Shape({-1, 784}));
    auto y = builder.input(ops::Shape({-1, 10}));

    auto l1 = dense_layer(x, 784, 100, relu);
    auto l2 = dense_layer(l1, 100, 10, nullptr);
    auto loss = softmax_cross_entropy(y, l2);

    SGDOptimizer optimizer(0.5 / 100);
    auto train_op = optimizer.minimize(loss);

    dbl_t loss_val;

    for (int i = 0; i < 100; ++i)
    {

        graph.run({train_op, loss},
                  {{x, {x_train, ops::Shape({100, 784})}},
                      {y, {y_train, ops::Shape({100, 10})}}},
                  {nullptr, &loss_val});

        std::cout << "epoch " << i << ", "
                  << "loss = " << loss_val << std::endl;
    }

    tensor_free(x_train);
    tensor_free(y_train);
}
