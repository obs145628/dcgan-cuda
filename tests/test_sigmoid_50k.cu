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

#include <tocha/tensor.hh>
#include "../src/datasets/mnist.hh"
#include "../src/memory/alloc.hh"

void linspace(dbl_t first, dbl_t last, std::size_t len, dbl_t* out)
{

    dbl_t coeff = (last - first) / (len - 1);
    for (std::size_t i = 0; i < len; ++i)
        out[i] = first + i * coeff;
}

int main(int argc, char** argv)
{

    if (argc != 2)
    {
        std::cerr << "Invalid number of arguments\n";
        return 1;
    }


    constexpr int N = 53147;
    dbl_t logits[N];
    linspace(0, 1, N, logits);
    

    auto& builder = ops::OpsBuilder::instance();
    
    auto x = builder.input(ops::Shape({N}));
    auto y = builder.vect_sigmoid(x);

    auto& graph = ops::Graph::instance();
    graph.debug_set(true);


    tocha::Tensors out;
    out.add(tocha::Tensor::f32(N));
    dbl_t* y_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);

        graph.run({y},
	      {{x, {logits, ops::Shape({N})}}},
	      {y_out});
    
    
    out.save(argv[1]);
}
