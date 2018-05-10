#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/sigmoid-cross-entropy.hh"
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


    dbl_t x[] = {
        1, -2.5, 0.4,
        0.2, -0.341, 0.7,
        -2.3, -12.5, 8.4,
        1.9, 1.2, 1.4,
        -0.23, -1.6, 1.4
    };

    dbl_t y[] = {
        0.1, 0.5, 0.4,
        0.2, 0.1, 0.7,
        0.8, 0.05, 0.15,
        0.3, 0.6, 0.1,
        0.7, 0.1, 0.2
    };


    auto& builder = ops::OpsBuilder::instance();
    
    auto x_node = builder.input(ops::Shape({5, 3}));
    auto y_node = builder.input(ops::Shape({5, 3}));
    auto loss_node = builder.sigmoid_cross_entropy(y_node, x_node); 

    auto& graph = ops::Graph::instance();


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(1));
    dbl_t* loss = reinterpret_cast<dbl_t*>(out.arr()[0].data);

    graph.run({loss_node},
              {{x_node, {x, ops::Shape({5, 3})}},
                  {y_node, {y, ops::Shape({5, 3})}}},
	      {loss});
    
    out.save(argv[1]);
}
