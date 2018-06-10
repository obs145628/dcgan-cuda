#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/log-softmax.hh"
#include "../src/ops/conv2d-bias-add.hh"
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
      14.0, 12.0, 7.0, -6.0,
      -5.0, -5.0, 9.0, -11.0,
      19.0, -1.0, -2.0, 2.0,
      7.0, 11.0, 12.0, -11.0};

     dbl_t bias[] = {2.0, -1.0};


    auto& builder = ops::OpsBuilder::instance();

    auto inputNode = builder.input(ops::Shape({2, 2, 2, 2}));
    auto biasNode = builder.input(ops::Shape({2}));
    auto convNode = builder.conv2d_bias_add(inputNode, biasNode);

    auto& graph = ops::Graph::instance();

    tocha::Tensors out;
    out.add(tocha::Tensor::f32(2, 2, 2, 2));
    dbl_t* conv_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);

        graph.run({convNode},
	      {
         {inputNode, {input, ops::Shape({2, 2, 2, 2})}},
         {biasNode, {bias, ops::Shape({2})}}
        },
	      {conv_out});


    out.save(argv[1]);
}