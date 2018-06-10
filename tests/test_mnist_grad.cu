#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/graph.hh"
#include "../src/api/activ.hh"
#include "../src/api/layers.hh"
#include "../src/api/cost.hh"
#include "../src/api/copy-initializer.hh"

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

    auto dx = graph.gradient(cost, x);
    auto dw1 = graph.gradient(cost, l1_data.w);
    auto db1 = graph.gradient(cost, l1_data.b);
    auto dl1 = graph.gradient(cost, l1);
    auto dw2 = graph.gradient(cost, l2_data.w);
    auto db2 = graph.gradient(cost, l2_data.b);
    auto dl2 = graph.gradient(cost, l2);

    tocha::Tensors out;
    out.add(tocha::Tensor::f32(20, 10));
    dbl_t* l2_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    out.add(tocha::Tensor::f32(1));
    dbl_t* loss_out = reinterpret_cast<dbl_t*>(out.arr()[1].data);
    out.add(tocha::Tensor::f32(20, 784));
    dbl_t* dx_out = reinterpret_cast<dbl_t*>(out.arr()[2].data);
    out.add(tocha::Tensor::f32(784, 100));
    dbl_t* dw1_out = reinterpret_cast<dbl_t*>(out.arr()[3].data);
    out.add(tocha::Tensor::f32(100));
    dbl_t* db1_out = reinterpret_cast<dbl_t*>(out.arr()[4].data);
    out.add(tocha::Tensor::f32(20, 100));
    dbl_t* dl1_out = reinterpret_cast<dbl_t*>(out.arr()[5].data);
    out.add(tocha::Tensor::f32(100, 10));
    dbl_t* dw2_out = reinterpret_cast<dbl_t*>(out.arr()[6].data);
    out.add(tocha::Tensor::f32(10));
    dbl_t* db2_out = reinterpret_cast<dbl_t*>(out.arr()[7].data);
    out.add(tocha::Tensor::f32(20, 10));
    dbl_t* dl2_out = reinterpret_cast<dbl_t*>(out.arr()[8].data);
    graph.run({l2, cost, dx, dw1, db1, dl1, dw2, db2, dl2},
	      {{x, {x_train, ops::Shape({20, 784})}},
		  {y, {y_train, ops::Shape({20, 10})}}},
	      {l2_out, loss_out, dx_out, dw1_out, db1_out, dl1_out,
                      dw2_out, db2_out, dl2_out});

    
    out.add(tocha::Tensor::f32(30, 10));
    dbl_t* l2_out2 = reinterpret_cast<dbl_t*>(out.arr()[9].data);
    out.add(tocha::Tensor::f32(1));
    dbl_t* loss_out2 = reinterpret_cast<dbl_t*>(out.arr()[10].data);
    out.add(tocha::Tensor::f32(30, 784));
    dbl_t* dx_out2 = reinterpret_cast<dbl_t*>(out.arr()[11].data);
    out.add(tocha::Tensor::f32(784, 100));
    dbl_t* dw1_out2 = reinterpret_cast<dbl_t*>(out.arr()[12].data);
    out.add(tocha::Tensor::f32(100));
    dbl_t* db1_out2 = reinterpret_cast<dbl_t*>(out.arr()[13].data);
    out.add(tocha::Tensor::f32(30, 100));
    dbl_t* dl1_out2 = reinterpret_cast<dbl_t*>(out.arr()[14].data);
    out.add(tocha::Tensor::f32(100, 10));
    dbl_t* dw2_out2 = reinterpret_cast<dbl_t*>(out.arr()[15].data);
    out.add(tocha::Tensor::f32(10));
    dbl_t* db2_out2 = reinterpret_cast<dbl_t*>(out.arr()[16].data);
    out.add(tocha::Tensor::f32(30, 10));
    dbl_t* dl2_out2 = reinterpret_cast<dbl_t*>(out.arr()[17].data);
    graph.run({l2, cost, dx, dw1, db1, dl1, dw2, db2, dl2},
	      {{x, {x_train + 20 * 784, ops::Shape({30, 784})}},
		  {y, {y_train + 20 * 10, ops::Shape({30, 10})}}},
	      {l2_out2, loss_out2, dx_out2, dw1_out2, db1_out2, dl1_out2,
                      dw2_out2, db2_out2, dl2_out2});

    out.add(tocha::Tensor::f32(30, 10));
    dbl_t* l2_out3 = reinterpret_cast<dbl_t*>(out.arr()[18].data);
    out.add(tocha::Tensor::f32(1));
    dbl_t* loss_out3 = reinterpret_cast<dbl_t*>(out.arr()[19].data);
    out.add(tocha::Tensor::f32(30, 784));
    dbl_t* dx_out3 = reinterpret_cast<dbl_t*>(out.arr()[20].data);
    out.add(tocha::Tensor::f32(784, 100));
    dbl_t* dw1_out3 = reinterpret_cast<dbl_t*>(out.arr()[21].data);
    out.add(tocha::Tensor::f32(100));
    dbl_t* db1_out3 = reinterpret_cast<dbl_t*>(out.arr()[22].data);
    out.add(tocha::Tensor::f32(30, 100));
    dbl_t* dl1_out3 = reinterpret_cast<dbl_t*>(out.arr()[23].data);
    out.add(tocha::Tensor::f32(100, 10));
    dbl_t* dw2_out3 = reinterpret_cast<dbl_t*>(out.arr()[24].data);
    out.add(tocha::Tensor::f32(10));
    dbl_t* db2_out3 = reinterpret_cast<dbl_t*>(out.arr()[25].data);
    out.add(tocha::Tensor::f32(30, 10));
    dbl_t* dl2_out3 = reinterpret_cast<dbl_t*>(out.arr()[26].data);
    graph.run({l2, cost, dx, dw1, db1, dl1, dw2, db2, dl2},
	      {{x, {x_train + 50 * 784, ops::Shape({30, 784})}},
		  {y, {y_train + 50 * 10, ops::Shape({30, 10})}}},
	      {l2_out3, loss_out3, dx_out3, dw1_out3, db1_out3, dl1_out3,
                      dw2_out3, db2_out3, dl2_out3});
    

    out.save(argv[3]);
    delete[] x_train;
    delete[] y_train;
}
