#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/softmax.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/mat-rvect-add.hh"
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

    dbl_t b[] = {
        1., 5., 3.
    };

    const int size = sizeof(a) / (3 * sizeof(dbl_t));

    auto& builder = ops::OpsBuilder::instance();

    auto a_node = builder.input(ops::Shape({size, 3}));
    auto b_node = builder.input(ops::Shape({3}));
    
    auto res_node = builder.mat_rvect_add(a_node, b_node);

    auto& graph = ops::Graph::instance();


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(size, 3));
    dbl_t* res = reinterpret_cast<dbl_t*>(out.arr()[0].data);

        graph.run({res_node},
                  {{a_node, {a, ops::Shape({size, 3})}},
                      {b_node, {b, ops::Shape({3})}}},
	      {res});
    
    
    out.save(argv[1]);
}
