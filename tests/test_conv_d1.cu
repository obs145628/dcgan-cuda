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
#include "../src/utils/xorshift.hh"

#include <tocha/tensor.hh>
#include "../src/datasets/mnist.hh"
#include "../src/memory/alloc.hh"

#define BATCH 1

int main(int argc, char** argv)
{

    if (argc != 2)
    {
        std::cerr << "Invalid number of arguments\n";
        return 1;
    }
    
    xorshift::seed(234);

    dbl_t x[BATCH * 32 * 32 * 64];
    dbl_t w[5 * 5 * 64 * 128];
    xorshift::fill(x, x + BATCH * 32 * 32 * 64);
    xorshift::fill(w, w + 5 * 5 * 64 * 128);


    auto& builder = ops::OpsBuilder::instance();

    auto x_node = builder.input(ops::Shape({BATCH, 32, 32, 64}));
    auto w_node = builder.input(ops::Shape({5, 5, 64, 128}));
    const int strides[2] = {2, 2};
    auto y_node = builder.conv2d(x_node, w_node, strides);

    auto& graph = ops::Graph::instance();


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(BATCH, 16, 16, 128));
    dbl_t* y_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);


    graph.run({y_node},
	      {
                  {x_node, {x, ops::Shape({BATCH, 32, 32, 64})}},
                  {w_node, {w, ops::Shape({5, 5, 64, 128})}}
              },
	      {y_out});

    out.save(argv[1]);
}
