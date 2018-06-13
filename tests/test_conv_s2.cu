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
#include "../src/ops/mse.hh"
#include "../src/ops/reshape.hh"

#include <tocha/tensor.hh>
#include "../src/datasets/mnist.hh"
#include "../src/memory/alloc.hh"

#define BATCH 64

int main(int argc, char** argv)
{

    if (argc != 2)
    {
        std::cerr << "Invalid number of arguments\n";
        return 1;
    }
    
    xorshift::seed(234);

    

    auto x = new dbl_t[BATCH * 64 * 64 * 3];
    auto w0 = new dbl_t[5 * 5 * 3 * 64];
    auto w1 = new dbl_t[5 * 5 * 64 * 128];
    auto w2 = new dbl_t[5 * 5 * 128 * 256];
    auto w3 = new dbl_t[5 * 5 * 256 * 512];
    auto y = new dbl_t[BATCH * 4 * 4 * 512];

    xorshift::fill(x, x + BATCH * 64 * 64 * 3);
    xorshift::fill(w0, w0 + 5 * 5 * 3 * 64);
    xorshift::fill(w1, w1 + 5 * 5 * 64 * 128);
    xorshift::fill(w2, w2 + 5 * 5 * 128 * 256);
    xorshift::fill(w3, w3 + 5 * 5 * 256 * 512);
    xorshift::fill(y, y + BATCH * 4 * 4 * 512);
    
    auto& builder = ops::OpsBuilder::instance();
    auto& graph = ops::Graph::instance();
    graph.debug_set(true);
    
    auto x_node = builder.input(ops::Shape({BATCH, 64, 64, 3}));
    auto w0_node = builder.input(ops::Shape({5, 5, 3, 64}));
    auto w1_node = builder.input(ops::Shape({5, 5, 64, 128}));
    auto w2_node = builder.input(ops::Shape({5, 5, 128, 256}));
    auto w3_node = builder.input(ops::Shape({5, 5, 256, 512}));
    const int strides[2] = {2, 2};
    ops::Op* yh_node = builder.conv2d(x_node, w0_node, strides);
    yh_node = builder.conv2d(yh_node, w1_node, strides);
    yh_node = builder.conv2d(yh_node, w2_node, strides);
    yh_node = builder.conv2d(yh_node, w3_node, strides);
    yh_node = builder.reshape(yh_node, ops::Shape({BATCH, 4 * 4 * 512}));

    auto y_node = builder.input(ops::Shape({BATCH, 4 * 4 * 512}));
    auto mse_node = builder.mse(y_node, yh_node);
    auto dx_node = graph.gradient(mse_node, x_node);


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(BATCH, 64, 64, 3));
    dbl_t* dx_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);


    graph.run({dx_node},
	      {
                  {x_node, {x, ops::Shape({BATCH, 64, 64, 3})}},
                  {w0_node, {w0, ops::Shape({5, 5, 3, 64})}},
                  {w1_node, {w1, ops::Shape({5, 5, 64, 128})}},
                  {w2_node, {w2, ops::Shape({5, 5, 128, 256})}},
                  {w3_node, {w3, ops::Shape({5, 5, 256, 512})}},
                  {y_node, {y, ops::Shape({BATCH, 4 * 4 * 512})}}
              },
	      {dx_out});

    out.save(argv[1]);

    delete[] x;
    delete[] w0;
    delete[] w1;
    delete[] w2;
    delete[] w3;
}
