#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/softmax.hh"
#include "../src/ops/mat-sum.hh"
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
	1, 2,
        3, 4,
        5, 6
    };


    auto& builder = ops::OpsBuilder::instance();
    
    auto x_node = builder.input(ops::Shape({3, 2}));
    auto y_node = builder.mat_sum(x_node, 1);

    auto& graph = ops::Graph::instance();


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(3));
    dbl_t* y = reinterpret_cast<dbl_t*>(out.arr()[0].data);

    graph.run({y_node},
	      {{x_node, {x, ops::Shape({3, 2})}}},
	      {y});
    
    
    out.save(argv[1]);
}
