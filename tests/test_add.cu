#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/softmax.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/graph.hh"
#include "../src/api/layers.hh"
#include "../src/api/cost.hh"
#include "../src/ops/add.hh"

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


    dbl_t a[] = {
	0.1, 1.2, 4.3,
	4.1, 0.2, 7.3,
	0.06, 2.01, 0.23,
	5.6, 2.3, 1.18
    };

    dbl_t b[] = { 
        2.1, -5, 4.2,
        4.1, 0.6, 8.2,
        102.76, 342.91, -2.23,
        -50.6, 56.23, 15.18
    };


    auto& builder = ops::OpsBuilder::instance();
    
    auto a_node = builder.input(ops::Shape({4, 3}));
    auto b_node = builder.input(ops::Shape({4, 3}));
    auto y_node = builder.add(a_node, b_node);

    auto& graph = ops::Graph::instance();


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(4, 3));
    dbl_t* y_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);

        graph.run({y_node},
                  {{a_node, {a, ops::Shape({4, 3})}},
                   {b_node, {b, ops::Shape({4, 3})}}},
	      {y_out});
    
    
    out.save(argv[1]);
}
