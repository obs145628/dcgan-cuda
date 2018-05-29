#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/softmax.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/mse.hh"
#include "../src/ops/argmax-accuracy.hh"
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

    dbl_t y_hat[] = {
        0.1, 0.2, 0.7,
        0.8, .1, .1,
        0.1, 0.3, 0.6,
        .6, .2, .2,
        .1, .1, .8,
        .2, .3, .5,
        .7, .1, .2,
        .4, .3, .3,
        .2, .1, .7,
        .8, .1, .1
    };

    dbl_t y[] = {
        0., 1, 0,
        0, 0, 1,
        1, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 0,
        0, 1, 0,
        1, 0, 0
    };


    auto& builder = ops::OpsBuilder::instance();

    auto y_node = builder.input(ops::Shape({10, 3}));
    auto y_hat_node = builder.input(ops::Shape({10, 3}));
    
    auto res_node = builder.argmax_accuracy(y_node, y_hat_node);

    auto& graph = ops::Graph::instance();


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(1));
    dbl_t* res = reinterpret_cast<dbl_t*>(out.arr()[0].data);

    graph.run({res_node},
              {{y_node, {y, ops::Shape({10, 3})}},
                  {y_hat_node, {y_hat, ops::Shape({10, 3})}}},
	      {res});
    
    
    out.save(argv[1]);
}
