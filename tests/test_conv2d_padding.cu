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
        5.0, 0.0, 4.0,
        -1.0, -1.0, 8.0,
        2.0, -2.0, 9.0,
        3.0, 4.0, -5.0,
        8.0, -4.0, 1.0,
        3.0, 3.0, 1.0,
        6.0, 2.0, -5.0,
        3.0, 4.0, -2.0,
        4.0, 0.0, 10.0,
        1.0, 2.0,-3.0,
        1.0, 0.0, 5.0,
        0.0, -2.0, 2.0,
        0.0, 1.0, -7.0,
        2.0, 0.0, -4.0,
        -3.0, -10.0, 3.0,
        8.0, 2.0, -4.0,
        0.0, 1.0, -1.0,
        4.0, -2.0, -8.0,
        1.0, -1.0, 2.0,
        -7.0, -4.0, -5.0,
        3.0, 0.0, 4.0,
        4.0, 2.0, 9.0,
        3.0, 0.0, -12.0,
        -3.0, 0.0, 4.0,
        6.0, -10.0, 12.0,
        1.0, 0.0, 5.0,
        -8.0, 7.0, 0.0,
        0.0, 1.0, 4.0,
        0.0, -9.0, 1.0,
        0.0, 1.0, 0.0};

     dbl_t kernel[] = {
        1.0, 1.0,
        -1.0, 1.0,
         0.0, 0.0,
         0.0, -1.0,
        -1.0, 1.0,
         1.0, 0.0,
        -1.0, -1.0,
         1.0, 0.0,
         0.0, 1.0,
         0.0, -1.0,
         0.0, 0.0,
         1.0, -1.0,
         1.0, 1.0,
        -1.0, 0.0,
         1.0, -1.0,
        -1.0, 0.0,
        -1.0, 0.0,
        -1.0, 1.0};


    auto& builder = ops::OpsBuilder::instance();

    auto inputNode = builder.input(ops::Shape({2, 3, 5, 3}));
    auto kernelNode = builder.input(ops::Shape({2, 3, 3, 2}));
    const int strides[2] = {2, 2};
    auto convNode = builder.conv2d(inputNode, kernelNode, strides);

    auto& graph = ops::Graph::instance();


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(2, 2, 3, 2));
    dbl_t* conv_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);

        graph.run({convNode},
	      {
         {inputNode, {input, ops::Shape({2, 3, 5, 3})}},
         {kernelNode, {kernel, ops::Shape({2, 3, 3, 2})}}
        },
	      {conv_out});


    out.save(argv[1]);
}
