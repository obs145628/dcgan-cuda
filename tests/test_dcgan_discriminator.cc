#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/sigmoid-cross-entropy.hh"
#include "../src/ops/reshape.hh"
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

    auto celeba =tocha::Tensors::load(argv[1]);
    auto x_train = reinterpret_cast<dbl_t*>(celeba.arr()[0].data);
    std::size_t len = celeba.arr()[0].dims[0];
    dbl_t* y_train = tensor_alloc(len);
    std::fill_n(y_train, len, 1);

    auto weights = tocha::Tensors::load(argv[2]);
    CopyInitializer w0(reinterpret_cast<dbl_t*>(weights.arr()[0].data));
    CopyInitializer b0(reinterpret_cast<dbl_t*>(weights.arr()[1].data));
    CopyInitializer w1(reinterpret_cast<dbl_t*>(weights.arr()[2].data));
    CopyInitializer b1(reinterpret_cast<dbl_t*>(weights.arr()[3].data));
    CopyInitializer w2(reinterpret_cast<dbl_t*>(weights.arr()[4].data));
    CopyInitializer b2(reinterpret_cast<dbl_t*>(weights.arr()[5].data));
    CopyInitializer w3(reinterpret_cast<dbl_t*>(weights.arr()[6].data));
    CopyInitializer b3(reinterpret_cast<dbl_t*>(weights.arr()[7].data));
    CopyInitializer w4(reinterpret_cast<dbl_t*>(weights.arr()[8].data));
    CopyInitializer b4(reinterpret_cast<dbl_t*>(weights.arr()[9].data));


    auto& graph = ops::Graph::instance();
    graph.debug_set(true);
    auto& builder = ops::OpsBuilder::instance();

    auto x = builder.input(ops::Shape({10, 64, 64, 3}));
    auto y = builder.input(ops::Shape({10, 1}));


    std::size_t kernel_size[] = {5, 5};
    int stride_size[] = {2, 2};

    std::size_t l0_size[] = {10, 64, 64, 3};
    Conv2DLayerData tmp_l0;
    auto l0 = conv2d_layer(x, 64, kernel_size, stride_size, l0_size, leaky_relu,
                           &w0, &b0, &tmp_l0);

    std::cout << l0->shape_get() << std::endl;

    Conv2DLayerData tmp_l1;
    std::size_t l1_size[] = {10, 32, 32, 64};
    auto l1 = conv2d_layer(l0, 128, kernel_size, stride_size, l1_size, nullptr,
                           &w1, &b1, &tmp_l1);

    Conv2DLayerData tmp_l2;
    std::size_t l2_size[] = {10, 16, 16, 128};
    auto l2 = conv2d_layer(l1, 256, kernel_size, stride_size, l2_size, nullptr,
                           &w2, &b2, &tmp_l2);

    Conv2DLayerData tmp_l3;
    std::size_t l3_size[] = {10, 8, 8, 256};
    auto l3 = conv2d_layer(l2, 512, kernel_size, stride_size, l3_size, nullptr,
                           &w3, &b3, &tmp_l3);

    DenseLayerData tmp_l4;
    ops::Op* l4 = builder.reshape(l3, ops::Shape({10, 4 * 4 * 512}));
    l4 = dense_layer(l4, 4 * 4 * 512, 1, nullptr, &w4, &b4, &tmp_l4);
    auto logits = l4;
    auto loss = builder.sigmoid_cross_entropy(y, logits);

    auto dw0 = graph.gradient(loss, tmp_l0.w);
    auto db0 = graph.gradient(loss, tmp_l0.b);
    auto dw1 = graph.gradient(loss, tmp_l1.w);
    auto db1 = graph.gradient(loss, tmp_l1.b);
    auto dw2 = graph.gradient(loss, tmp_l2.w);
    auto db2 = graph.gradient(loss, tmp_l2.b);
    auto dw3 = graph.gradient(loss, tmp_l3.w);
    auto db3 = graph.gradient(loss, tmp_l3.b);
    auto dw4 = graph.gradient(loss, tmp_l4.w);
    auto db4 = graph.gradient(loss, tmp_l4.b);

    tocha::Tensors out;

    out.add(tocha::Tensor::f32(10, 16, 16, 128));
    dbl_t* l1_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);

    out.add(tocha::Tensor::f32(10, 4, 4, 512));
    dbl_t* l3_out = reinterpret_cast<dbl_t*>(out.arr()[1].data);

    out.add(tocha::Tensor::f32(10, 1));
    dbl_t* logits_out = reinterpret_cast<dbl_t*>(out.arr()[2].data);

    out.add(tocha::Tensor::f32(1));
    dbl_t* loss_out = reinterpret_cast<dbl_t*>(out.arr()[3].data);

    out.add(tocha::Tensor::f32(5, 5, 3, 64));
    dbl_t* dw0_out = reinterpret_cast<dbl_t*>(out.arr()[4].data);
    out.add(tocha::Tensor::f32(64));
    dbl_t* db0_out = reinterpret_cast<dbl_t*>(out.arr()[5].data);
    out.add(tocha::Tensor::f32(5, 5, 64, 128));

    dbl_t* dw1_out = reinterpret_cast<dbl_t*>(out.arr()[6].data);
    out.add(tocha::Tensor::f32(128));
    dbl_t* db1_out = reinterpret_cast<dbl_t*>(out.arr()[7].data);
    out.add(tocha::Tensor::f32(5, 5, 128, 256));
    dbl_t* dw2_out = reinterpret_cast<dbl_t*>(out.arr()[8].data);
    out.add(tocha::Tensor::f32(256));
    dbl_t* db2_out = reinterpret_cast<dbl_t*>(out.arr()[9].data);
    out.add(tocha::Tensor::f32(5, 5, 256, 512));
    dbl_t* dw3_out = reinterpret_cast<dbl_t*>(out.arr()[10].data);
    out.add(tocha::Tensor::f32(512));
    dbl_t* db3_out = reinterpret_cast<dbl_t*>(out.arr()[11].data);
    out.add(tocha::Tensor::f32(8192, 1));
    dbl_t* dw4_out = reinterpret_cast<dbl_t*>(out.arr()[12].data);
    out.add(tocha::Tensor::f32(1));
    dbl_t* db4_out = reinterpret_cast<dbl_t*>(out.arr()[13].data);

    graph.run({l1, l3, logits, loss, dw0, db0, dw1, db1, dw2, db2, dw3, db3, dw4, db4},
	      {{x, {x_train, ops::Shape({10, 64, 64, 3})}},
		  {y, {y_train, ops::Shape({10, 1})}}},
	      {l1_out, l3_out, logits_out, loss_out,
                      dw0_out, db0_out, dw1_out, db1_out, dw2_out, db2_out,
                      dw3_out, db3_out, dw4_out, db4_out});

    out.save(argv[3]);
    tensor_free(y_train);
}
