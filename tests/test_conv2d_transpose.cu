#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/log-softmax.hh"
#include "../src/ops/conv2d-transpose.hh"
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
      -0.125, -2.25,
      -1.75, -0.25,
      -0.5, 3.0,
      0.25, -0.375
    };

     dbl_t kernel[] = {
        1.0, -1.0, -1.0, 1.0
     };


    auto& builder = ops::OpsBuilder::instance();

    auto inputNode = builder.input(ops::Shape({2, 2, 2, 1}));
    auto kernelNode = builder.input(ops::Shape({2, 2, 1, 1}));
    const int strides[2] = {2, 2};
    const int out_size[4] = {2, 4, 4, 1};
    auto convNode = builder.conv2d_transpose(inputNode, kernelNode, out_size,
                                             strides);

    auto& graph = ops::Graph::instance();


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(2, 4, 4, 1));
    dbl_t* conv_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);

        graph.run({convNode},
	      {
         {inputNode, {input, ops::Shape({2, 2, 2, 1})}},
         {kernelNode, {kernel, ops::Shape({2, 2, 1, 1})}}
        },
	      {conv_out});


    out.save(argv[1]);
}
