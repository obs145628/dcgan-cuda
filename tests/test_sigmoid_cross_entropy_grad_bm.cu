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

#include "big_mat.hh"

int main(int argc, char** argv)
{

    if (argc != 2)
    {
        std::cerr << "Invalid number of arguments\n";
        return 1;
    }

    const int size = sizeof(a) / (3 * sizeof(dbl_t));
    
    auto& graph = ops::Graph::instance();
    auto& builder = ops::OpsBuilder::instance();
    
    auto x_node = builder.input(ops::Shape({size, 3}));
    auto y_node = builder.input(ops::Shape({size, 3}));
    auto loss_node = builder.sigmoid_cross_entropy(y_node, x_node); 
    auto dx_node = graph.gradient(loss_node, x_node);
    


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(1));
    dbl_t* loss = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    out.add(tocha::Tensor::f32(size, 3));
    dbl_t* dx = reinterpret_cast<dbl_t*>(out.arr()[1].data);

    graph.run({loss_node, dx_node},
              {{x_node, {a, ops::Shape({size, 3})}},
                  {y_node, {a, ops::Shape({size, 3})}}},
	      {loss, dx});
    
    out.save(argv[1]);
}
