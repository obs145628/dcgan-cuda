#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-relu-leaky.hh"
#include "../src/ops/vect-relu.hh"
#include "../src/ops/softmax.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
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

    const int size = sizeof(v) / sizeof(dbl_t);

    auto& builder = ops::OpsBuilder::instance();

    auto x = builder.input(ops::Shape({1, size}));
    auto y = builder.vect_relu_leaky(x);

    auto& graph = ops::Graph::instance();

    tocha::Tensors out;
    out.add(tocha::Tensor::f32(1, size));
    dbl_t* y_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);

    graph.run({y},
	      {{x, {v, ops::Shape({1, size})}}},
	      {y_out});


    out.save(argv[1]);
}
