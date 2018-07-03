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
#include "../src/ops/update.hh"
#include "../src/ops/moment-update.hh"
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

    dbl_t a[] = {
        1, 2,
        3, 4
    };

    dbl_t b[] = {
        10, 30,
        20, 40
    };

    dbl_t coeff = 5.7;


    auto& graph = ops::Graph::instance();
    auto& builder = ops::OpsBuilder::instance();
    auto a_node = builder.variable(ops::Shape({2, 2}));
    auto b_node = builder.input(ops::Shape({2, 2}));
    auto coeff_node  = builder.input(ops::Shape());
    auto res_node = builder.update(a_node, b_node, coeff_node);

    a_node->write(a);

    graph.run({res_node},
              {{b_node, {b, ops::Shape({2, 2})}},
                  {coeff_node, {&coeff, ops::Shape()}}},
	      {nullptr});

    tocha::Tensors out;
    out.add(tocha::Tensor::f32(2, 2));
    dbl_t* res = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    a_node->read(res);
    
    out.save(argv[1]);
}
