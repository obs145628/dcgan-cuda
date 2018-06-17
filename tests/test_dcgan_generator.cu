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
#include "../src/api/normal-initializer.hh"
#include "../src/ops/conv2d-transpose.hh"
#include "../src/ops/conv2d-bias-add.hh"

#define BATCH 64
#define Z_DIM 100

int g_strides[] = {2, 2};
std::size_t g_kernel_size[] = {5, 5};

ops::Op* conv2d_transpose(ops::Op* input,
                          const int* in_size,
                          const int* out_size,
                          bool use_activ,
                          Initializer* w_init,
                          Initializer* b_init,
                          Conv2DTransposeLayerData* tmp_data)
{
    auto& builder = ops::OpsBuilder::instance();

    auto w = builder.variable(ops::Shape({int(g_kernel_size[0]), int(g_kernel_size[1]),
                                          int(out_size[3]), int(in_size[3])}), true);
    w->extend_name("conv2d_transpose_w");
    auto b = builder.variable(ops::Shape({int(out_size[3])}), true);
    b->extend_name("conv2d_transpose_b");

    NormalInitializer base_init;
    if (!w_init)
        w_init = &base_init;
    if (!b_init)
        b_init = &base_init;

    w_init->fill(w->data_begin(), w->data_end());
    b_init->fill(b->data_begin(), b->data_end());

    ops::Op* out_conv = builder.conv2d_transpose(input, w, out_size, g_strides);
    ops::Op* out = builder.conv2d_bias_add(out_conv, b);
    ops::Op* z = out;
    if (use_activ)
    {
        out = leaky_relu(z);
        out->extend_name("conv2d_transpose_activ");
    }

    if (tmp_data)
    {
        tmp_data->w = w;
        tmp_data->b = b;
        tmp_data->z = z;
    }

    return out;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
	std::cerr << "Invalid number of arguments\n";
	return 1;
    }

    auto celeba =tocha::Tensors::load(argv[1]);
    auto x_train = reinterpret_cast<dbl_t*>(celeba.arr()[0].data);
    std::size_t len = BATCH * 64 * 64 * 3;
    dbl_t* y_train = new dbl_t[len];
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

    auto x = builder.input(ops::Shape({BATCH, Z_DIM}));
    auto y = builder.input(ops::Shape({BATCH, 64 * 64 * 3}));

    
    int l0_size[] = {BATCH, 4, 4, 512};
    int l1_size[] = {BATCH, 8, 8, 256};
    int l2_size[] = {BATCH, 16, 16, 128};
    int l3_size[] = {BATCH, 32, 32, 64};
    int l4_size[] = {BATCH, 64, 64, 3};

    DenseLayerData tmp_l0;
    ops::Op* l0 = dense_layer(x, Z_DIM, 4 * 4 * 512, leaky_relu, &w0, &b0, &tmp_l0);
    l0 = builder.reshape(l0, ops::Shape({BATCH, 4, 4, 512}));
    
    Conv2DTransposeLayerData tmp_l1;
    ops::Op* l1 = conv2d_transpose(l0, l0_size, l1_size, true, &w1, &b1, &tmp_l1);
    std::cout << l1->shape_get() << std::endl;
    
    Conv2DTransposeLayerData tmp_l2;
    ops::Op* l2 = conv2d_transpose(l1, l1_size, l2_size, true, &w2, &b2, &tmp_l2);
    std::cout << l2->shape_get() << std::endl;

    Conv2DTransposeLayerData tmp_l3;
    ops::Op* l3 = conv2d_transpose(l2, l2_size, l3_size, true, &w3, &b3, &tmp_l3);

    Conv2DTransposeLayerData tmp_l4;
    ops::Op* l4 = conv2d_transpose(l3, l3_size, l4_size, false, &w4, &b4, &tmp_l4);

    auto logits = l4;
    ops::Op* logits_flat = builder.reshape(logits, ops::Shape({BATCH, 64 * 64 * 3}));
    auto loss = builder.sigmoid_cross_entropy(y, logits_flat);

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

    out.add(tocha::Tensor::f32(BATCH, 4, 4, 512));
    dbl_t* l0_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    
    out.add(tocha::Tensor::f32(BATCH, 8, 8, 256));
    dbl_t* l1_out = reinterpret_cast<dbl_t*>(out.arr()[1].data);

    out.add(tocha::Tensor::f32(BATCH, 32, 32, 64));
    dbl_t* l3_out = reinterpret_cast<dbl_t*>(out.arr()[2].data);

    out.add(tocha::Tensor::f32(BATCH, 64, 64, 3));
    dbl_t* logits_out = reinterpret_cast<dbl_t*>(out.arr()[3].data);

    out.add(tocha::Tensor::f32(1));
    dbl_t* loss_out = reinterpret_cast<dbl_t*>(out.arr()[4].data);

    out.add(tocha::Tensor::f32(Z_DIM, 4 * 4 * 512));
    dbl_t* dw0_out = reinterpret_cast<dbl_t*>(out.arr()[5].data);
    out.add(tocha::Tensor::f32(4 * 4 * 512));
    dbl_t* db0_out = reinterpret_cast<dbl_t*>(out.arr()[6].data);

    out.add(tocha::Tensor::f32(5, 5, 256, 512));
    dbl_t* dw1_out = reinterpret_cast<dbl_t*>(out.arr()[7].data);
    out.add(tocha::Tensor::f32(256));
    dbl_t* db1_out = reinterpret_cast<dbl_t*>(out.arr()[8].data);
    out.add(tocha::Tensor::f32(5, 5, 128, 256));
    dbl_t* dw2_out = reinterpret_cast<dbl_t*>(out.arr()[9].data);
    out.add(tocha::Tensor::f32(128));
    dbl_t* db2_out = reinterpret_cast<dbl_t*>(out.arr()[10].data);
    out.add(tocha::Tensor::f32(5, 5, 64, 128));
    dbl_t* dw3_out = reinterpret_cast<dbl_t*>(out.arr()[11].data);
    out.add(tocha::Tensor::f32(64));
    dbl_t* db3_out = reinterpret_cast<dbl_t*>(out.arr()[12].data);
    out.add(tocha::Tensor::f32(5, 5, 3, 64));
    dbl_t* dw4_out = reinterpret_cast<dbl_t*>(out.arr()[13].data);
    out.add(tocha::Tensor::f32(3));
    dbl_t* db4_out = reinterpret_cast<dbl_t*>(out.arr()[14].data);

    graph.run({l0, l1, l3, logits, loss, dw0, db0, dw1, db1, dw2, db2, dw3, db3, dw4, db4},
	      {{x, {x_train, ops::Shape({BATCH, Z_DIM})}},
		  {y, {y_train, ops::Shape({BATCH, 64 * 64 * 3})}}},
	      {l0_out, l1_out, l3_out, logits_out, loss_out,
                      dw0_out, db0_out, dw1_out, db1_out, dw2_out, db2_out,
                      dw3_out, db3_out, dw4_out, db4_out});

    out.save(argv[3]);
    delete[] y_train;
}
