#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/graph.hh"
#include "../src/api/activ.hh"
#include "../src/api/layers.hh"
#include "../src/api/cost.hh"

#include <tocha/tensor.hh>
#include "../src/datasets/mnist.hh"
#include "../src/memory/alloc.hh"


int main(int argc, char** argv)
{
    if (argc != 4)
    {
	std::cerr << "Invalid number of arguments\n";
	return 1;
    }

    dbl_t* x_train;
    dbl_t* y_train;
    mnist::load(argv[1], &x_train, &y_train);
    auto weights = tocha::Tensors::load(argv[2]);

    dbl_t* w1 = reinterpret_cast<dbl_t*>(weights.arr()[0].data);
    dbl_t* b1 = reinterpret_cast<dbl_t*>(weights.arr()[1].data);
    dbl_t* w2 = reinterpret_cast<dbl_t*>(weights.arr()[2].data);
    dbl_t* b2 = reinterpret_cast<dbl_t*>(weights.arr()[3].data);


    auto& graph = ops::Graph::instance();
    graph.debug_set(true);
    auto& builder = ops::OpsBuilder::instance();
    
    auto x = builder.input(ops::Shape({-1, 784}));
    auto y = builder.input(ops::Shape({-1, 10}));

    auto l1 = dense_layer(x, 784, 100, sigmoid, w1, b1);
    auto l2 = dense_layer(l1, 100, 10, sigmoid, w2, b2);
    auto y_hat = l2;
    auto cost = quadratic_cost(y, y_hat);

    tocha::Tensors out;
    out.add(tocha::Tensor::f32(20, 10));
    dbl_t* y_hat_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    out.add(tocha::Tensor::f32(1));
    dbl_t* loss_out = reinterpret_cast<dbl_t*>(out.arr()[1].data);
    graph.run({y_hat, cost},
	      {{x, {x_train, ops::Shape({20, 784})}},
		  {y, {y_train, ops::Shape({20, 10})}}},
	      {y_hat_out, loss_out});


    out.add(tocha::Tensor::f32(30, 10));
    dbl_t* y_hat_out2 = reinterpret_cast<dbl_t*>(out.arr()[2].data);
    out.add(tocha::Tensor::f32(1));
    dbl_t* loss_out2 = reinterpret_cast<dbl_t*>(out.arr()[3].data);
    graph.run({y_hat, cost},
	      {{x, {x_train + 20 * 784, ops::Shape({30, 784})}},
		  {y, {y_train + 20 * 10, ops::Shape({30, 10})}}},
	      {y_hat_out2, loss_out2});

    out.add(tocha::Tensor::f32(30, 10));
    dbl_t* y_hat_out3 = reinterpret_cast<dbl_t*>(out.arr()[4].data);
    out.add(tocha::Tensor::f32(1));
    dbl_t* loss_out3 = reinterpret_cast<dbl_t*>(out.arr()[5].data);
    graph.run({y_hat, cost},
	      {{x, {x_train + 50 * 784, ops::Shape({30, 784})}},
		  {y, {y_train + 50 * 10, ops::Shape({30, 10})}}},
	      {y_hat_out3, loss_out3});
    

    out.save(argv[3]);
    tensor_free(x_train);
    tensor_free(y_train);
}
