#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/softmax.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/mat-mul-add.hh"
#include "../src/ops/graph.hh"
#include "../src/ops/mse.hh"
#include "../src/api/layers.hh"
#include "../src/api/cost.hh"

#include <tocha/tensor.hh>
#include "../src/datasets/mnist.hh"
#include "../src/memory/alloc.hh"

int main(int argc, char** argv)
{

    if (argc != 2)
    {
        std::cerr << "Invalid number of arguments\n";
        return 1;
    }    

    dbl_t x[] = {
        1., 2, 4,
        4.1, 0.5, 7,
        2, 2, 8,
        5, 2.3, 1.1
    };

    dbl_t w[] = {
        1., 5.,
        2., 4,
        3, 8
    };

    dbl_t b[] = {0.5, -4.6};

    dbl_t y[] = {
        0.1, 1.2,
        4.1, 0.2,
        0.06, 2.01,
        5.6, 2.3
    };


    auto& graph = ops::Graph::instance();
    auto& builder = ops::OpsBuilder::instance();

    auto x_node = builder.input(ops::Shape({4, 3}));
    auto w_node = builder.input(ops::Shape({3, 2}));
    auto b_node = builder.input(ops::Shape({2}));
    auto y_node = builder.input(ops::Shape({4, 2}));
    auto y_hat_node = builder.mat_mul_add(x_node, w_node, b_node);
    auto loss_node = builder.mse(y_node, y_hat_node);

    auto dx_node = graph.gradient(loss_node, x_node);
    auto dw_node = graph.gradient(loss_node, w_node);
    //auto db_node = graph.gradient(loss_node, b_node);
    auto dy_hat_node = graph.gradient(loss_node, y_hat_node);


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(4, 3));
    dbl_t* dx = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    out.add(tocha::Tensor::f32(3, 2));
    dbl_t* dw = reinterpret_cast<dbl_t*>(out.arr()[1].data);
    out.add(tocha::Tensor::f32(4, 2));
    dbl_t* dy_hat = reinterpret_cast<dbl_t*>(out.arr()[2].data);

    graph.run({dx_node, dw_node, dy_hat_node},
              {{x_node, {x, ops::Shape({4, 3})}},
                  {w_node, {w, ops::Shape({3, 2})}},
                  {b_node, {b, ops::Shape({2})}},
                  {y_node, {y, ops::Shape({4, 2})}}},
	      {dx, dw, dy_hat});
    
    
    out.save(argv[1]);
}
