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
#include "../src/ops/conv2d.hh"
#include "../src/api/zero-initializer.hh"
#include "../src/ops/mat-mul-add.hh"

#include <tocha/tensor.hh>
#include "../src/datasets/mnist.hh"
#include "../src/memory/alloc.hh"
#include "../src/api/normal-initializer.hh"
#include "../src/ops/conv2d-transpose.hh"
#include "../src/ops/conv2d-bias-add.hh"
#include "../src/ops/input.hh"
#include "../src/ops/vect-tanh.hh"
#include "../src/ops/add.hh"

#define BATCH 10
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


ops::Op* conv2d(ops::Op* input,
                std::size_t nb_filter,
                std::size_t* in_size,
                Initializer* w_init,
                Initializer* b_init,
                Conv2DLayerData* tmp_data,
                bool reuse)
{
    auto& builder = ops::OpsBuilder::instance();

    ops::Op* w = nullptr;
    ops::Op* b = nullptr;

    if (reuse)
    {
        w = tmp_data->w;
        b = tmp_data->b;
    }

    else
    {
        auto new_w = builder.variable(ops::Shape({int(g_kernel_size[0]), int(g_kernel_size[1]),
                                              int(in_size[3]), int(nb_filter)}), true);
        new_w->extend_name("conv2d_w");
        auto new_b = builder.variable(ops::Shape({int(nb_filter)}), true);
        new_b->extend_name("conv2d_b");
        
        NormalInitializer base_init;
        if (!w_init)
            w_init = &base_init;
        if (!b_init)
            b_init = &base_init;
        
        w_init->fill(new_w->data_begin(), new_w->data_end());
        b_init->fill(new_b->data_begin(), new_b->data_end());

        w = new_w;
        b = new_b;
    }

    ops::Op* out_conv = builder.conv2d(input, w, g_strides);
    ops::Op* out = builder.conv2d_bias_add(out_conv, b);
    ops::Op* z = out;
    out = leaky_relu(z);
    out->extend_name("conv2d_activ");


    tmp_data->w = w;
    tmp_data->b = b;
    tmp_data->z = z;
    return out;
}

ops::Op* dense(ops::Op* input,
               std::size_t in_size,
               std::size_t out_size,
               Initializer* w_init,
               Initializer* b_init,
               DenseLayerData* tmp_data,
               bool reuse)
{
    auto& builder = ops::OpsBuilder::instance();

    ops::Op* w = nullptr;
    ops::Op* b = nullptr;
    
    if (reuse)
    {
        w = tmp_data->w;
        b = tmp_data->b;
    }

    else
    {
        auto new_w = builder.variable(ops::Shape({int(in_size), int(out_size)}), true);
        new_w->extend_name("dense_w");
        auto new_b = builder.variable(ops::Shape({int(out_size)}), true);
        new_b->extend_name("dense_b");

        NormalInitializer w_base_init;
        ZeroInitializer b_base_init;
        if (!w_init)
            w_init = &w_base_init;
        if (!b_init)
            b_init = &b_base_init;

        w_init->fill(new_w->data_begin(), new_w->data_end());
        b_init->fill(new_b->data_begin(), new_b->data_end());

        w = new_w;
        b = new_b;
    }

    ops::Op* out = builder.mat_mul_add(input, w, b);
    ops::Op* z = out;

    tmp_data->w = w;
    tmp_data->b = b;
    tmp_data->z = z;
    return out;
}

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

    bool reuse = !layers.empty();
    auto& builder = ops::OpsBuilder::instance();
    if (!reuse)
        layers.resize(4);
    
    auto& tmp_l0 = layers[0];
    auto l0 = conv2d(x, 64, g_dl0_size, &w0, &b0, &tmp_l0, reuse);

    auto& tmp_l1 = layers[1];
    auto l1 = conv2d(l0, 128, g_dl1_size, &w1, &b1, &tmp_l1, reuse);

    auto& tmp_l2 = layers[2];
    auto l2 = conv2d(l1, 256, g_dl2_size, &w2, &b2, &tmp_l2, reuse);

    auto& tmp_l3 = layers[3];
    auto l3 = conv2d(l2, 512, g_dl3_size, &w3, &b3, &tmp_l3, reuse);

    //DenseLayerData tmp_l4;
    ops::Op* l4 = builder.reshape(l3, ops::Shape({BATCH, 4 * 4 * 512}));
    l4 = dense(l4, 4 * 4 * 512, 1, &w4, &b4, &layer_4, reuse);
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
    auto x_data = reinterpret_cast<dbl_t*>(celeba.arr()[0].data);

    dbl_t* logits_0 = new dbl_t[BATCH];
    dbl_t* logits_1 = new dbl_t[BATCH];
    std::fill_n(logits_0, BATCH, 0);
    std::fill_n(logits_1, BATCH, 1);
    auto logits0 = builder.input(ops::Shape({BATCH, 1}));
    auto logits1 = builder.input(ops::Shape({BATCH, 1}));

    auto weights = tocha::Tensors::load(argv[2]);


    auto z = builder.input(ops::Shape({BATCH, Z_DIM}));
    auto x = builder.input(ops::Shape({BATCH, 64, 64, 3}));


    DenseLayerData g_layer0;
    std::vector<Conv2DTransposeLayerData> g_layers;
    std::vector<ops::Op*> g_nodes;
    auto g_logits = generator(z, weights, g_layer0, g_layers, g_nodes);
    auto g_out = builder.vect_tanh(g_logits);

    DenseLayerData d_layer4;
    std::vector<Conv2DLayerData> d_layers;
    std::vector<ops::Op*> d1_nodes;
    auto d1_logits = discriminator(g_out, weights, d_layer4, d_layers, d1_nodes);
    std::vector<ops::Op*> d2_nodes;
    auto d2_logits = discriminator(x, weights, d_layer4, d_layers, d2_nodes);

    auto g_loss = builder.sigmoid_cross_entropy(logits1, d1_logits);
    auto d1_loss = builder.sigmoid_cross_entropy(logits0, d1_logits);
    auto d2_loss = builder.sigmoid_cross_entropy(logits1, d2_logits);

    auto d_loss = builder.add(d1_loss, d2_loss);
    
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


    out.add(tocha::Tensor::f32(Z_DIM, 4 * 4 * 512));
    dbl_t* g_dw0_out = reinterpret_cast<dbl_t*>(out.arr()[12].data);
    out.add(tocha::Tensor::f32(4 * 4 * 512));
    dbl_t* g_db0_out = reinterpret_cast<dbl_t*>(out.arr()[13].data);

    out.add(tocha::Tensor::f32(5, 5, 256, 512));
    dbl_t* g_dw1_out = reinterpret_cast<dbl_t*>(out.arr()[14].data);
    out.add(tocha::Tensor::f32(256));
    dbl_t* g_db1_out = reinterpret_cast<dbl_t*>(out.arr()[15].data);
    out.add(tocha::Tensor::f32(5, 5, 128, 256));
    dbl_t* g_dw2_out = reinterpret_cast<dbl_t*>(out.arr()[16].data);
    out.add(tocha::Tensor::f32(128));
    dbl_t* g_db2_out = reinterpret_cast<dbl_t*>(out.arr()[17].data);
    out.add(tocha::Tensor::f32(5, 5, 64, 128));
    dbl_t* g_dw3_out = reinterpret_cast<dbl_t*>(out.arr()[18].data);
    out.add(tocha::Tensor::f32(64));
    dbl_t* g_db3_out = reinterpret_cast<dbl_t*>(out.arr()[19].data);
    out.add(tocha::Tensor::f32(5, 5, 3, 64));
    dbl_t* g_dw4_out = reinterpret_cast<dbl_t*>(out.arr()[20].data);
    out.add(tocha::Tensor::f32(3));
    dbl_t* g_db4_out = reinterpret_cast<dbl_t*>(out.arr()[21].data);

    out.add(tocha::Tensor::f32(5, 5, 3, 64));
    dbl_t* d_dw0_out = reinterpret_cast<dbl_t*>(out.arr()[22].data);
    out.add(tocha::Tensor::f32(64));
    dbl_t* d_db0_out = reinterpret_cast<dbl_t*>(out.arr()[23].data);
    out.add(tocha::Tensor::f32(5, 5, 64, 128));
    dbl_t* d_dw1_out = reinterpret_cast<dbl_t*>(out.arr()[24].data);
    out.add(tocha::Tensor::f32(128));
    dbl_t* d_db1_out = reinterpret_cast<dbl_t*>(out.arr()[25].data);
    out.add(tocha::Tensor::f32(5, 5, 128, 256));
    dbl_t* d_dw2_out = reinterpret_cast<dbl_t*>(out.arr()[26].data);
    out.add(tocha::Tensor::f32(256));
    dbl_t* d_db2_out = reinterpret_cast<dbl_t*>(out.arr()[27].data);
    out.add(tocha::Tensor::f32(5, 5, 256, 512));
    dbl_t* d_dw3_out = reinterpret_cast<dbl_t*>(out.arr()[28].data);
    out.add(tocha::Tensor::f32(512));
    dbl_t* d_db3_out = reinterpret_cast<dbl_t*>(out.arr()[29].data);
    out.add(tocha::Tensor::f32(8192, 1));
    dbl_t* d_dw4_out = reinterpret_cast<dbl_t*>(out.arr()[30].data);
    out.add(tocha::Tensor::f32(1));
    dbl_t* d_db4_out = reinterpret_cast<dbl_t*>(out.arr()[31].data);

    graph.run({g_nodes[0], g_nodes[1], g_nodes[2], g_nodes[3], g_logits, g_out,
               d1_nodes[0], d1_nodes[1], d1_nodes[2], d1_nodes[3], d1_logits, g_loss,
               g_dw0, g_db0, g_dw1, g_db1, g_dw2, g_db2, g_dw3, g_db3, g_dw4, g_db4,
               d_dw0, d_db0, d_dw1, d_db1, d_dw2, d_db2, d_dw3, d_db3, d_dw4, d_db4},
	      {{z, {x_data, ops::Shape({BATCH, Z_DIM})}},
               {x, {x_data, ops::Shape({BATCH, 64, 64, 3})}},
               {logits0, {logits_0, ops::Shape({BATCH, 1})}},
               {logits1, {logits_1, ops::Shape({BATCH, 1})}},
              },
	      {g_l0_out, g_l1_out, g_l2_out, g_l3_out, g_logits_out, g_out_out,
               d_l0_out, d_l1_out, d_l2_out, d_l3_out, d_logits_out, g_loss_out,
               g_dw0_out, g_db0_out, g_dw1_out, g_db1_out, g_dw2_out, g_db2_out,
               g_dw3_out, g_db3_out, g_dw4_out, g_db4_out,
               d_dw0_out, d_db0_out, d_dw1_out, d_db1_out, d_dw2_out, d_db2_out,
               d_dw3_out, d_db3_out, d_dw4_out, d_db4_out
              });
    
    out.save(argv[3]);
    delete[] logits_0;
    delete[] logits_1;
}
