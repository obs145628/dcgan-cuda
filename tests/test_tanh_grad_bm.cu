#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-tanh.hh"
#include "../src/ops/softmax.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/mse.hh"
#include "../src/ops/graph.hh"
#include "../src/api/layers.hh"
#include "../src/api/cost.hh"

#include <tocha/tensor.hh>
#include "../src/datasets/mnist.hh"
#include "../src/memory/alloc.hh"

#include "big_mat.hh"

int main(int argc, char** argv)
{

    if (argc != 2)
    {
        std::cerr << "Invalid number of arguments\n";
        return 1;
    }

    const int size = sizeof(a) / (3 * sizeof(dbl_t));

    auto& graph = ops::Graph::instance();

    auto& builder = ops::OpsBuilder::instance();

    auto x_node = builder.input(ops::Shape({size, 3}));
    auto y_node = builder.input(ops::Shape({size, 3}));
    auto y_hat_node = builder.vect_tanh(x_node);
    
    auto loss_node = builder.mse(y_node, y_hat_node);
    auto dx_node = graph.gradient(loss_node, x_node);
    auto dy_hat_node = graph.gradient(loss_node, y_hat_node);


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(size, 3));
    dbl_t* y_hat = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    out.add(tocha::Tensor::f32(1));
    dbl_t* loss = reinterpret_cast<dbl_t*>(out.arr()[1].data);
    out.add(tocha::Tensor::f32(size, 3));
    dbl_t* dx = reinterpret_cast<dbl_t*>(out.arr()[2].data);
    out.add(tocha::Tensor::f32(size, 3));
    dbl_t* dy_hat = reinterpret_cast<dbl_t*>(out.arr()[3].data);

    graph.run({y_hat_node, loss_node, dx_node, dy_hat_node},
              {{x_node, {a, ops::Shape({size, 3})}},
                  {y_node, {a, ops::Shape({size, 3})}}},
	      {y_hat, loss, dx, dy_hat});
    
    
    out.save(argv[1]);
}
