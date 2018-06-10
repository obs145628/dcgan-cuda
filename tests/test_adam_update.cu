#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/softmax.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/adam-update.hh"
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

    dbl_t x[] = {
        1, 2,
        3, 4
    };

    dbl_t m[] = {
        10, 30,
        20, 40
    };

    dbl_t v[] = {
        4, 8,
        16, 12
    };

    auto& graph = ops::Graph::instance();
    auto& builder = ops::OpsBuilder::instance();
    auto x_node = builder.variable(ops::Shape({2, 2}));
    auto m_node = builder.input(ops::Shape({2, 2}));
    auto v_node = builder.input(ops::Shape({2, 2}));
    auto res_node = builder.adam_update(x_node, m_node, v_node, 0.45, 0.68, 0.27, 2);

    x_node->write(x);

    graph.run({res_node},
              {{m_node, {m, ops::Shape({2, 2})}},
                  {v_node, {v, ops::Shape({2, 2})}}},
	      {nullptr});

    tocha::Tensors out;
    out.add(tocha::Tensor::f32(2, 2));
    dbl_t* res = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    x_node->read(res);
    
    out.save(argv[1]);
}
