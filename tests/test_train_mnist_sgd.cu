#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/graph.hh"
#include "../src/api/activ.hh"
#include "../src/api/layers.hh"
#include "../src/api/cost.hh"
#include "../src/api/copy-initializer.hh"
#include "../src/api/sgd-optimizer.hh"

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

    CopyInitializer w1(reinterpret_cast<dbl_t*>(weights.arr()[0].data));
    CopyInitializer b1(reinterpret_cast<dbl_t*>(weights.arr()[1].data));
    CopyInitializer w2(reinterpret_cast<dbl_t*>(weights.arr()[2].data));
    CopyInitializer b2(reinterpret_cast<dbl_t*>(weights.arr()[3].data));


    auto& graph = ops::Graph::instance();
    graph.debug_set(true);
    auto& builder = ops::OpsBuilder::instance();
    
    auto x = builder.input(ops::Shape({-1, 784}));
    auto y = builder.input(ops::Shape({-1, 10}));

    DenseLayerData l1_data;
    auto l1 = dense_layer(x, 784, 100, relu, &w1, &b1, &l1_data);
    DenseLayerData l2_data;
    auto l2 = dense_layer(l1, 100, 10, nullptr, &w2, &b2, &l2_data);
    auto cost = softmax_cross_entropy(y, l2);

    auto sgd = SGDOptimizer(0.01);
    auto train_op = sgd.minimize(cost);

    tocha::Tensors out;
    out.add(tocha::Tensor::f32(784, 100));
    dbl_t* w1_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    out.add(tocha::Tensor::f32(100));
    dbl_t* b1_out = reinterpret_cast<dbl_t*>(out.arr()[1].data);
    out.add(tocha::Tensor::f32(100, 10));
    dbl_t* w2_out = reinterpret_cast<dbl_t*>(out.arr()[2].data);
    out.add(tocha::Tensor::f32(10));
    dbl_t* b2_out = reinterpret_cast<dbl_t*>(out.arr()[3].data);

    graph.run({train_op},
	      {{x, {x_train, ops::Shape({20, 784})}},
		  {y, {y_train, ops::Shape({20, 10})}}},
	      {nullptr});

    (dynamic_cast<ops::Variable*>(l1_data.w))->read(w1_out);
    (dynamic_cast<ops::Variable*>(l1_data.b))->read(b1_out);
    (dynamic_cast<ops::Variable*>(l2_data.w))->read(w2_out);
    (dynamic_cast<ops::Variable*>(l2_data.b))->read(b2_out);

    dbl_t loss_out;

    graph.run({cost},
	      {{x, {x_train, ops::Shape({20, 784})}},
		  {y, {y_train, ops::Shape({20, 10})}}},
	      {&loss_out});

    std::cout << loss_out << std::endl;
    
    

    out.save(argv[3]);
    tensor_free(x_train);
    tensor_free(y_train);
}
