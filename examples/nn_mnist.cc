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

    std::size_t batch_size = 100;

    std::vector<dbl_t> x_batch_vect(784 * batch_size);
    std::vector<dbl_t> y_batch_vect(10 * batch_size);
    dbl_t* x_batch = &x_batch_vect[0];
    dbl_t* y_batch = &y_batch_vect[0];

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

    SGDOptimizer optimizer(0.05);
    auto train_op = optimizer.minimize(loss);

    dbl_t loss_val;

    for (int i = 0; i < 1000; ++i)
    {

        for (std::size_t i = 0; i < batch_size; ++i)
        {
            int n = rand() % 70000;

            std::copy(x_train + 784 * n, x_train + 784 * (n + 1), x_batch + 784 * i);
            std::copy(y_train + 10 * n, y_train + 10 * (n + 1), y_batch + 10 * i);
        }
        

        graph.run({train_op, loss},
                  {{x, {x_batch, ops::Shape({100, 784})}},
                      {y, {y_batch, ops::Shape({100, 10})}}},
                  {nullptr, &loss_val});

        std::cout << "epoch " << i << ", "
                  << "loss = " << loss_val << std::endl;
    }

    tensor_free(x_train);
    tensor_free(y_train);
}
