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
#include "../src/ops/input.hh"
#include "../src/ops/vect-tanh.hh"

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
        out = relu(z);
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

int g_gl0_size[] = {BATCH, 4, 4, 512};
int g_gl1_size[] = {BATCH, 8, 8, 256};
int g_gl2_size[] = {BATCH, 16, 16, 128};
int g_gl3_size[] = {BATCH, 32, 32, 64};
int g_gl4_size[] = {BATCH, 64, 64, 3};

ops::Op* generator(ops::Op* x, const tocha::Tensors& weights,
                   DenseLayerData& layer_0,
                   std::vector<Conv2DTransposeLayerData>& layers,
                   std::vector<ops::Op*>& nodes)
{
    auto& builder = ops::OpsBuilder::instance();
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

    layers.resize(4);
    
    //DenseLayerData tmp_l0;
    ops::Op* l0 = dense_layer(x, Z_DIM, 4 * 4 * 512, relu, &w0, &b0, &layer_0);
    l0 = builder.reshape(l0, ops::Shape({BATCH, 4, 4, 512}));

    
    auto& tmp_l1 = layers[0];
    ops::Op* l1 = conv2d_transpose(l0, g_gl0_size, g_gl1_size, true, &w1, &b1, &tmp_l1);
    
    auto& tmp_l2 = layers[1];
    ops::Op* l2 = conv2d_transpose(l1, g_gl1_size, g_gl2_size, true, &w2, &b2, &tmp_l2);
    
    auto& tmp_l3 = layers[2];
    ops::Op* l3 = conv2d_transpose(l2, g_gl2_size, g_gl3_size, true, &w3, &b3, &tmp_l3);

    auto& tmp_l4 = layers[3];
    ops::Op* l4 = conv2d_transpose(l3, g_gl3_size, g_gl4_size, false, &w4, &b4, &tmp_l4);

    auto logits = l4;

    nodes.push_back(l0);
    nodes.push_back(l1);
    nodes.push_back(l2);
    nodes.push_back(l3);
    nodes.push_back(l4);
    return logits;
}

/*
ops::Op* conv2d_layer(ops::Op* input,
                      std::size_t nb_filter,
                      std::size_t* kernel_size,
                      int* strides,
                      std::size_t* in_size,
                      activ_f activ,
                      Initializer* w_init,
                      Initializer* b_init,
                      Conv2DLayerData* tmp_data)
{
    auto& builder = ops::OpsBuilder::instance();

    auto w = builder.variable(ops::Shape({int(kernel_size[0]), int(kernel_size[1]),
                                          int(in_size[3]), int(nb_filter)}), true);
    w->extend_name("conv2d_w");
    auto b = builder.variable(ops::Shape({int(nb_filter)}), true);
    b->extend_name("conv2d_b");

    NormalInitializer base_init;
    if (!w_init)
        w_init = &base_init;
    if (!b_init)
        b_init = &base_init;

    w_init->fill(w->data_begin(), w->data_end());
    b_init->fill(b->data_begin(), b->data_end());

    ops::Op* out_conv = builder.conv2d(input, w, strides);
    ops::Op* out = builder.conv2d_bias_add(out_conv, b);
    ops::Op* z = out;
    if (activ)
    {
        out = activ(z);
        out->extend_name("conv2d_activ");
    }

    if (tmp_data)
    {
        tmp_data->w = w;
        tmp_data->b = b;
        tmp_data->z = z;
    }

    return out;
}
*/

std::size_t g_dl0_size[] = {BATCH, 64, 64, 3};
std::size_t g_dl1_size[] = {BATCH, 32, 32, 64};
std::size_t g_dl2_size[] = {BATCH, 16, 16, 128};
std::size_t g_dl3_size[] = {BATCH, 8, 8, 256};

ops::Op* discriminator(ops::Op* x, const tocha::Tensors& weights,
                       DenseLayerData& layer_4,
                       std::vector<Conv2DLayerData>& layers,
                       std::vector<ops::Op*>& nodes)
{

    CopyInitializer w0(reinterpret_cast<dbl_t*>(weights.arr()[10].data));
    CopyInitializer b0(reinterpret_cast<dbl_t*>(weights.arr()[11].data));
    CopyInitializer w1(reinterpret_cast<dbl_t*>(weights.arr()[12].data));
    CopyInitializer b1(reinterpret_cast<dbl_t*>(weights.arr()[13].data));
    CopyInitializer w2(reinterpret_cast<dbl_t*>(weights.arr()[14].data));
    CopyInitializer b2(reinterpret_cast<dbl_t*>(weights.arr()[15].data));
    CopyInitializer w3(reinterpret_cast<dbl_t*>(weights.arr()[16].data));
    CopyInitializer b3(reinterpret_cast<dbl_t*>(weights.arr()[17].data));
    CopyInitializer w4(reinterpret_cast<dbl_t*>(weights.arr()[18].data));
    CopyInitializer b4(reinterpret_cast<dbl_t*>(weights.arr()[19].data));

    auto& builder = ops::OpsBuilder::instance();
    layers.resize(4);
    
    auto& tmp_l0 = layers[0];
    auto l0 = conv2d_layer(x, 64, g_kernel_size, g_strides, g_dl0_size, leaky_relu,
                           &w0, &b0, &tmp_l0);
    

    auto& tmp_l1 = layers[1];
    auto l1 = conv2d_layer(l0, 128, g_kernel_size, g_strides, g_dl1_size, leaky_relu,
                           &w1, &b1, &tmp_l1);

    auto& tmp_l2 = layers[2];
    auto l2 = conv2d_layer(l1, 256, g_kernel_size, g_strides, g_dl2_size, leaky_relu,
                           &w2, &b2, &tmp_l2);

    auto& tmp_l3 = layers[3];
    auto l3 = conv2d_layer(l2, 512, g_kernel_size, g_strides, g_dl3_size, leaky_relu,
                           &w3, &b3, &tmp_l3);

    //DenseLayerData tmp_l4;
    ops::Op* l4 = builder.reshape(l3, ops::Shape({BATCH, 4 * 4 * 512}));
    l4 = dense_layer(l4, 4 * 4 * 512, 1, nullptr, &w4, &b4, &layer_4);
    auto logits = l4;

    nodes.push_back(l0);
    nodes.push_back(l1);
    nodes.push_back(l2);
    nodes.push_back(l3);
    nodes.push_back(l4);
    return logits;
}


int main(int argc, char** argv)
{
    if (argc != 4)
    {
	std::cerr << "Invalid number of arguments\n";
	return 1;
    }

    auto& graph = ops::Graph::instance();
    graph.debug_set(true);
    auto& builder = ops::OpsBuilder::instance();

    auto celeba = tocha::Tensors::load(argv[1]);
    auto x_train = reinterpret_cast<dbl_t*>(celeba.arr()[0].data);

    dbl_t* logits_0 = new dbl_t[BATCH];
    dbl_t* logits_1 = new dbl_t[BATCH];
    std::fill_n(logits_0, BATCH, 0);
    std::fill_n(logits_1, BATCH, 1);
    auto logits0 = builder.input(ops::Shape({BATCH, 1}));
    auto logits1 = builder.input(ops::Shape({BATCH, 1}));

    auto weights = tocha::Tensors::load(argv[2]);


    auto x = builder.input(ops::Shape({BATCH, Z_DIM}));


    DenseLayerData g_layer0;
    std::vector<Conv2DTransposeLayerData> g_layers;
    std::vector<ops::Op*> g_nodes;
    auto g_logits = generator(x, weights, g_layer0, g_layers, g_nodes);
    auto g_out = builder.vect_tanh(g_logits);

    DenseLayerData d_layer4;
    std::vector<Conv2DLayerData> d_layers;
    std::vector<ops::Op*> d1_nodes;
    auto d1_logits = discriminator(g_out, weights, d_layer4, d_layers, d1_nodes);

    auto g_loss = builder.sigmoid_cross_entropy(logits1, d1_logits);
    auto d1_loss = builder.sigmoid_cross_entropy(logits0, d1_logits);

    auto d_loss = d1_loss;
    
    auto g_dw0 = graph.gradient(g_loss, g_layer0.w);
    auto g_db0 = graph.gradient(g_loss, g_layer0.b);
    auto g_dw1 = graph.gradient(g_loss, g_layers[0].w);
    auto g_db1 = graph.gradient(g_loss, g_layers[0].b);
    auto g_dw2 = graph.gradient(g_loss, g_layers[1].w);
    auto g_db2 = graph.gradient(g_loss, g_layers[1].b);
    auto g_dw3 = graph.gradient(g_loss, g_layers[2].w);
    auto g_db3 = graph.gradient(g_loss, g_layers[2].b);
    auto g_dw4 = graph.gradient(g_loss, g_layers[3].w);
    auto g_db4 = graph.gradient(g_loss, g_layers[3].b);

    auto d_dw0 = graph.gradient(d_loss, d_layers[0].w);
    auto d_db0 = graph.gradient(d_loss, d_layers[0].b);
    auto d_dw1 = graph.gradient(d_loss, d_layers[1].w);
    auto d_db1 = graph.gradient(d_loss, d_layers[1].b);
    auto d_dw2 = graph.gradient(d_loss, d_layers[2].w);
    auto d_db2 = graph.gradient(d_loss, d_layers[2].b);
    auto d_dw3 = graph.gradient(d_loss, d_layers[3].w);
    auto d_db3 = graph.gradient(d_loss, d_layers[3].b);
    auto d_dw4 = graph.gradient(d_loss, d_layer4.w);
    auto d_db4 = graph.gradient(d_loss, d_layer4.b);

    tocha::Tensors out;

    out.add(tocha::Tensor::f32(BATCH, 4, 4, 512));
    dbl_t* g_l0_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    
    out.add(tocha::Tensor::f32(BATCH, 8, 8, 256));
    dbl_t* g_l1_out = reinterpret_cast<dbl_t*>(out.arr()[1].data);

    out.add(tocha::Tensor::f32(BATCH, 16, 16, 128));
    dbl_t* g_l2_out = reinterpret_cast<dbl_t*>(out.arr()[2].data);

    out.add(tocha::Tensor::f32(BATCH, 32, 32, 64));
    dbl_t* g_l3_out = reinterpret_cast<dbl_t*>(out.arr()[3].data);

    out.add(tocha::Tensor::f32(BATCH, 64, 64, 3));
    dbl_t* g_logits_out = reinterpret_cast<dbl_t*>(out.arr()[4].data);;

    out.add(tocha::Tensor::f32(BATCH, 64, 64, 3));
    dbl_t* g_out_out = reinterpret_cast<dbl_t*>(out.arr()[5].data);

    out.add(tocha::Tensor::f32(BATCH, 32, 32, 64));
    dbl_t* d_l0_out = reinterpret_cast<dbl_t*>(out.arr()[6].data);

    out.add(tocha::Tensor::f32(BATCH, 16, 16, 128));
    dbl_t* d_l1_out = reinterpret_cast<dbl_t*>(out.arr()[7].data);

    out.add(tocha::Tensor::f32(BATCH, 8, 8, 256));
    dbl_t* d_l2_out = reinterpret_cast<dbl_t*>(out.arr()[8].data);

    out.add(tocha::Tensor::f32(BATCH, 4, 4, 512));
    dbl_t* d_l3_out = reinterpret_cast<dbl_t*>(out.arr()[9].data);

    out.add(tocha::Tensor::f32(BATCH, 1));
    dbl_t* d_logits_out = reinterpret_cast<dbl_t*>(out.arr()[10].data);
    
    out.add(tocha::Tensor::f32(1));
    dbl_t* g_loss_out = reinterpret_cast<dbl_t*>(out.arr()[11].data);

    graph.run({g_nodes[0], g_nodes[1], g_nodes[2], g_nodes[3], g_logits, g_out,
               d1_nodes[0], d1_nodes[1], d1_nodes[2], d1_nodes[3], d1_logits, g_loss},
	      {{x, {x_train, ops::Shape({BATCH, Z_DIM})}},
               //{logits0, {logits_0, ops::Shape({BATCH, 1})}},
               {logits1, {logits_1, ops::Shape({BATCH, 1})}},
              },
	      {g_l0_out, g_l1_out, g_l2_out, g_l3_out, g_logits_out, g_out_out,
               d_l0_out, d_l1_out, d_l2_out, d_l3_out, d_logits_out, g_loss_out});

    out.save(argv[3]);
    delete[] logits_0;
    delete[] logits_1;
}
