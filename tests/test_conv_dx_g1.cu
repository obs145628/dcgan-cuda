#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/log-softmax.hh"
#include "../src/ops/conv2d.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/graph.hh"
#include "../src/api/layers.hh"
#include "../src/api/cost.hh"
#include "../src/ops/reshape.hh"
#include "../src/ops/mse.hh"
#include "../src/ops/conv2d-transpose.hh"

#include <tocha/tensor.hh>
#include "../src/datasets/mnist.hh"
#include "../src/memory/alloc.hh"

#define BATCH 64

int main(int argc, char** argv)
{

    if (argc < 4)
    {
        std::cerr << "Invalid number of arguments\n";
        return 1;
    }

    
    
    auto weights = tocha::Tensors::load(argv[2]);
    auto x = reinterpret_cast<dbl_t*>(weights.arr()[0].data);
    auto w = reinterpret_cast<dbl_t*>(weights.arr()[1].data);
    auto y = reinterpret_cast<dbl_t*>(weights.arr()[2].data);

    auto& builder = ops::OpsBuilder::instance();
    auto& graph = ops::Graph::instance();
    
    auto x_node = builder.input(ops::Shape({BATCH, 8, 8, 256}));
    auto w_node = builder.input(ops::Shape({5, 5, 128, 256}));
    auto y_node = builder.input(ops::Shape({BATCH, 16 * 16 * 128}));
    const int strides[2] = {2, 2};
    
    const int out_size[4] = {BATCH, 16, 16, 128};
    ops::Op* yh_node = builder.conv2d_transpose(x_node, w_node, out_size, strides);
    yh_node = builder.reshape(yh_node, ops::Shape({BATCH, 16 * 16 * 128}));

    auto mse_node = builder.mse(y_node, yh_node);
    auto dx_node = graph.gradient(mse_node, x_node);


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(BATCH, 8, 8, 256));
    dbl_t* dx_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);


    graph.run({dx_node},
	      {
                  {x_node, {x, ops::Shape({BATCH, 8, 8, 256})}},
                  {w_node, {w, ops::Shape({5, 5, 128, 256})}},
                  {y_node, {y, ops::Shape({BATCH, 16 * 16 * 128})}},
              },
	      {dx_out});

    out.save(argv[3]);
}
