#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/softmax.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/reshape.hh"
#include "../src/ops/graph.hh"
#include "../src/ops/mse.hh"
#include "../src/api/layers.hh"
#include "../src/api/cost.hh"
#include "../src/ops/conv2d-bias-add.hh"

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
    dbl_t input[] = {
            9.0, 13.0, 2.0, 12.0,
            -8.0, 9.0, 3.0, -9.0,
            11.0, 3.0, 0.0, -22.0,
            3.0, 9.0, 18.0, -8.0
    };

    dbl_t bias[] = {1.0, 2.0};

    dbl_t y[] = {
        7.0, 11.0, -1.0, 4.0,
        -6.0, 3.0, 5.0, 4.0,
        15.0, 4.0, -2.0, -22.0,
        0.0, 9.0, 22.0, -6.0
    };
    auto& graph = ops::Graph::instance();
    graph.debug_set(true);
    auto& builder = ops::OpsBuilder::instance();

    auto x_node = builder.input(ops::Shape({2, 2, 2, 2}));
    auto b_node = builder.input(ops::Shape({2}));
    auto y_node = builder.input(ops::Shape({2, 2, 2, 2}));

    auto y_node_reshape = builder.reshape(y_node, ops::Shape({2, 8}));
    auto y_hat_node = builder.conv2d_bias_add(x_node, b_node);
    auto y_hat_node_reshape = builder.reshape(y_hat_node, ops::Shape({2, 8}));
    auto loss_node_shape = builder.mse(y_node_reshape, y_hat_node_reshape);

    auto dy_hat_node = graph.gradient(loss_node_shape, y_hat_node_reshape);
    auto dx_node = graph.gradient(loss_node_shape, x_node);
    auto db_node = graph.gradient(loss_node_shape, b_node);

    tocha::Tensors out;
    out.add(tocha::Tensor::f32(2, 2, 2, 2));
    dbl_t* dx = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    out.add(tocha::Tensor::f32(2));
    dbl_t* db = reinterpret_cast<dbl_t*>(out.arr()[1].data);
    out.add(tocha::Tensor::f32(2, 2, 2, 2));
    dbl_t* dy_hat = reinterpret_cast<dbl_t*>(out.arr()[2].data);

    graph.run({dx_node, db_node, dy_hat_node},
              {{x_node, {input, ops::Shape({2, 2, 2, 2})}},
               {b_node, {bias, ops::Shape({2})}},
               {y_node, {y, ops::Shape({2, 2, 2, 2})}}},
	      {dx, db, dy_hat});

    out.save(argv[1]);
}
