#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/softmax-cross-entropy.hh"
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


    dbl_t y[] = {
	0.1, 0.2, 0.7,
	0.8, .1, .1,
	0.1, 0.3, 0.6,
	.6, .2, .2
    };

    dbl_t logits[] = {
	0.1, 1.2, 4.3,
	4.1, 0.2, 7.3,
	0.06, 2.01, 0.23,
	5.6, 2.3, 1.18
    };


    auto& builder = ops::OpsBuilder::instance();
    
    auto y_node = builder.input(ops::Shape({4, 3}));
    auto logits_node = builder.input(ops::Shape({4, 3}));
    auto loss_node = builder.softmax_cross_entropy(y_node, logits_node); 

    auto& graph = ops::Graph::instance();


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(1));
    dbl_t* loss = reinterpret_cast<dbl_t*>(out.arr()[0].data);

        graph.run({loss_node},
		  {{y_node, {y, ops::Shape({4, 3})}},
		      {logits_node, {logits, ops::Shape({4, 3})}}},
	      {loss});
    
    out.save(argv[1]);
}
