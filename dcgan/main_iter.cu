#include <algorithm>
#include <random>
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
#include "../src/datasets/celeba.hh"
#include "../src/utils/arguments.hh"
#include "../src/api/adam-optimizer.hh"
#include "../src/api/sgd-optimizer.hh"
#include "../src/memory/copy.hh"

#include <tocha/tensor.hh>
#include "../src/datasets/mnist.hh"
#include "../src/memory/alloc.hh"
#include "../src/api/normal-initializer.hh"
#include "../src/ops/conv2d-transpose.hh"
#include "../src/ops/conv2d-bias-add.hh"
#include "../src/ops/input.hh"
#include "../src/ops/vect-tanh.hh"
#include "../src/ops/add.hh"

#define DATASET_LEN (202599) //number of images in the celebA dataset
#define BATCH (10)
#define SAMPLE_SIZE (9)
#define Z_DIM (100) //size of the noise input vector to generate image
#define LEARNING_RATE (0.0002) //adam optimizer learning rate
#define BETA1 (0.5) //adam optimizer beta1 parameter

int g_strides[] = {2, 2};
std::size_t g_kernel_size[] = {5, 5};

ops::Op* conv2d_transpose(ops::Op* input,
                          const int* in_size,
                          const int* out_size,
                          Initializer& w_init,
                          Initializer& b_init,
                          bool use_activ,
                          Conv2DTransposeLayerData* tmp_data)
{
    auto& builder = ops::OpsBuilder::instance();

    auto w = builder.variable(ops::Shape({int(g_kernel_size[0]), int(g_kernel_size[1]),
                                          int(out_size[3]), int(in_size[3])}), true);
    w->extend_name("conv2d_transpose_w");
    auto b = builder.variable(ops::Shape({int(out_size[3])}), true);
    b->extend_name("conv2d_transpose_b");

    w_init.fill(w->data_begin(), w->data_end());
    b_init.fill(b->data_begin(), b->data_end());

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

ops::Op* generator(ops::Op* x,
                   DenseLayerData& layer_0,
                   std::vector<Conv2DTransposeLayerData>& layers,
                   std::vector<ops::Op*>& nodes,
                   const tocha::Tensors& weights)
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
    
    ops::Op* l0 = dense_layer(x, Z_DIM, 4 * 4 * 512, relu, &w0, &b0, &layer_0);
    l0 = builder.reshape(l0, ops::Shape({BATCH, 4, 4, 512}));
    
    auto& tmp_l1 = layers[0];
    ops::Op* l1 = conv2d_transpose(l0, g_gl0_size, g_gl1_size, w1, b1, true, &tmp_l1);
    
    auto& tmp_l2 = layers[1];
    ops::Op* l2 = conv2d_transpose(l1, g_gl1_size, g_gl2_size, w2, b2, true, &tmp_l2);
    
    auto& tmp_l3 = layers[2];
    ops::Op* l3 = conv2d_transpose(l2, g_gl2_size, g_gl3_size, w3, b3, true, &tmp_l3);

    auto& tmp_l4 = layers[3];
    ops::Op* l4 = conv2d_transpose(l3, g_gl3_size, g_gl4_size, w4, b4, false, &tmp_l4);

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
                Initializer& w_init,
                Initializer& b_init,
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
        
        w_init.fill(new_w->data_begin(), new_w->data_end());
        b_init.fill(new_b->data_begin(), new_b->data_end());

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
               Initializer& w_init,
               Initializer& b_init,
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

        w_init.fill(new_w->data_begin(), new_w->data_end());
        b_init.fill(new_b->data_begin(), new_b->data_end());

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

ops::Op* discriminator(ops::Op* x,
                       DenseLayerData& layer_4,
                       std::vector<Conv2DLayerData>& layers,
                       std::vector<ops::Op*>& nodes,
                       const tocha::Tensors& weights)
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
    auto l0 = conv2d(x, 64, g_dl0_size, w0, b0, &tmp_l0, reuse);

    auto& tmp_l1 = layers[1];
    auto l1 = conv2d(l0, 128, g_dl1_size, w1, b1, &tmp_l1, reuse);

    auto& tmp_l2 = layers[2];
    auto l2 = conv2d(l1, 256, g_dl2_size, w2, b2, &tmp_l2, reuse);

    auto& tmp_l3 = layers[3];
    auto l3 = conv2d(l2, 512, g_dl3_size, w3, b3, &tmp_l3, reuse);

    ops::Op* l4 = builder.reshape(l3, ops::Shape({BATCH, 4 * 4 * 512}));
    l4 = dense(l4, 4 * 4 * 512, 1, w4, b4, &layer_4, reuse);
    auto logits = l4;

    nodes.push_back(l0);
    nodes.push_back(l1);
    nodes.push_back(l2);
    nodes.push_back(l3);
    nodes.push_back(l4);
    return logits;
}

void save_weights(const std::string& path, const std::vector<ops::Variable*>& g_vars,
                  const std::vector<ops::Variable*>& d_vars)
{
    tocha::Tensors wout;

    wout.add(tocha::Tensor::f32(Z_DIM, 4 * 4 * 512));
    dbl_t* g_w0_out = reinterpret_cast<dbl_t*>(wout.arr()[0].data);
    wout.add(tocha::Tensor::f32(4 * 4 * 512));
    dbl_t* g_b0_out = reinterpret_cast<dbl_t*>(wout.arr()[1].data);
    wout.add(tocha::Tensor::f32(5, 5, 256, 512));
    dbl_t* g_w1_out = reinterpret_cast<dbl_t*>(wout.arr()[2].data);
    wout.add(tocha::Tensor::f32(256));
    dbl_t* g_b1_out = reinterpret_cast<dbl_t*>(wout.arr()[3].data);
    wout.add(tocha::Tensor::f32(5, 5, 128, 256));
    dbl_t* g_w2_out = reinterpret_cast<dbl_t*>(wout.arr()[4].data);
    wout.add(tocha::Tensor::f32(128));
    dbl_t* g_b2_out = reinterpret_cast<dbl_t*>(wout.arr()[5].data);
    wout.add(tocha::Tensor::f32(5, 5, 64, 128));
    dbl_t* g_w3_out = reinterpret_cast<dbl_t*>(wout.arr()[6].data);
    wout.add(tocha::Tensor::f32(64));
    dbl_t* g_b3_out = reinterpret_cast<dbl_t*>(wout.arr()[7].data);
    wout.add(tocha::Tensor::f32(5, 5, 3, 64));
    dbl_t* g_w4_out = reinterpret_cast<dbl_t*>(wout.arr()[8].data);
    wout.add(tocha::Tensor::f32(3));
    dbl_t* g_b4_out = reinterpret_cast<dbl_t*>(wout.arr()[9].data);
    wout.add(tocha::Tensor::f32(5, 5, 3, 64));
    dbl_t* d_w0_out = reinterpret_cast<dbl_t*>(wout.arr()[10].data);
    wout.add(tocha::Tensor::f32(64));
    dbl_t* d_b0_out = reinterpret_cast<dbl_t*>(wout.arr()[11].data);
    wout.add(tocha::Tensor::f32(5, 5, 64, 128));
    dbl_t* d_w1_out = reinterpret_cast<dbl_t*>(wout.arr()[12].data);
    wout.add(tocha::Tensor::f32(128));
    dbl_t* d_b1_out = reinterpret_cast<dbl_t*>(wout.arr()[13].data);
    wout.add(tocha::Tensor::f32(5, 5, 128, 256));
    dbl_t* d_w2_out = reinterpret_cast<dbl_t*>(wout.arr()[14].data);
    wout.add(tocha::Tensor::f32(256));
    dbl_t* d_b2_out = reinterpret_cast<dbl_t*>(wout.arr()[15].data);
    wout.add(tocha::Tensor::f32(5, 5, 256, 512));
    dbl_t* d_w3_out = reinterpret_cast<dbl_t*>(wout.arr()[16].data);
    wout.add(tocha::Tensor::f32(512));
    dbl_t* d_b3_out = reinterpret_cast<dbl_t*>(wout.arr()[17].data);
    wout.add(tocha::Tensor::f32(8192, 1));
    dbl_t* d_w4_out = reinterpret_cast<dbl_t*>(wout.arr()[18].data);
    wout.add(tocha::Tensor::f32(1));
    dbl_t* d_b4_out = reinterpret_cast<dbl_t*>(wout.arr()[19].data);

    
    tensor_read(g_vars[0]->data_begin(), g_vars[0]->data_end(), g_w1_out);
    tensor_read(g_vars[1]->data_begin(), g_vars[1]->data_end(), g_b1_out);
    tensor_read(g_vars[2]->data_begin(), g_vars[2]->data_end(), g_w2_out);
    tensor_read(g_vars[3]->data_begin(), g_vars[3]->data_end(), g_b2_out);
    tensor_read(g_vars[4]->data_begin(), g_vars[4]->data_end(), g_w3_out);
    tensor_read(g_vars[5]->data_begin(), g_vars[5]->data_end(), g_b3_out);
    tensor_read(g_vars[6]->data_begin(), g_vars[6]->data_end(), g_w4_out);
    tensor_read(g_vars[7]->data_begin(), g_vars[7]->data_end(), g_b4_out);
    tensor_read(g_vars[8]->data_begin(), g_vars[8]->data_end(), g_w0_out);
    tensor_read(g_vars[9]->data_begin(), g_vars[9]->data_end(), g_b0_out);
    tensor_read(d_vars[0]->data_begin(), d_vars[0]->data_end(), d_w0_out);
    tensor_read(d_vars[1]->data_begin(), d_vars[1]->data_end(), d_b0_out);
    tensor_read(d_vars[2]->data_begin(), d_vars[2]->data_end(), d_w1_out);
    tensor_read(d_vars[3]->data_begin(), d_vars[3]->data_end(), d_b1_out);
    tensor_read(d_vars[4]->data_begin(), d_vars[4]->data_end(), d_w2_out);
    tensor_read(d_vars[5]->data_begin(), d_vars[5]->data_end(), d_b2_out);
    tensor_read(d_vars[6]->data_begin(), d_vars[6]->data_end(), d_w3_out);
    tensor_read(d_vars[7]->data_begin(), d_vars[7]->data_end(), d_b3_out);
    tensor_read(d_vars[8]->data_begin(), d_vars[8]->data_end(), d_w4_out);
    tensor_read(d_vars[9]->data_begin(), d_vars[9]->data_end(), d_b4_out);
    wout.save(path);
}

void generate_samples(const std::string& path, ops::Op* g_out, ops::Input* z, dbl_t* z_data = nullptr)
{
    auto& graph = ops::Graph::instance();
    dbl_t* generated = new dbl_t[BATCH * 64 * 64 * 3];

    bool is_null = !z_data;
    if (z_data == nullptr)
    {
        dbl_t* z_data = new dbl_t[SAMPLE_SIZE * Z_DIM];
        NormalInitializer init(0, 1);
        for (std::size_t i = 0; i < SAMPLE_SIZE * Z_DIM; ++i)
            z_data[i] = init.next();
    }
    
    graph.run({g_out},
              {{z, {z_data, ops::Shape({SAMPLE_SIZE, Z_DIM})}}},
              {generated});
    celeba::save_samples(generated, 3, 3, path);

    delete[] generated;
    if (is_null)
        delete[] z_data;
    std::cout << "Samples generated\n";
}

int main(int argc, char** argv)
{
    Arguments args(argc, argv);
    

    auto& graph = ops::Graph::instance();
    graph.debug_set(true);
    auto& builder = ops::OpsBuilder::instance();


    auto weights_in = tocha::Tensors::load("weights0.npz");
    
    auto z = builder.input(ops::Shape({BATCH, Z_DIM}));
    auto x = builder.input(ops::Shape({BATCH, 64, 64, 3}));

    DenseLayerData g_layer0;
    std::vector<Conv2DTransposeLayerData> g_layers;
    std::vector<ops::Op*> g_nodes;
    auto g_logits = generator(z, g_layer0, g_layers, g_nodes, weights_in);
    auto g_out = builder.vect_tanh(g_logits);

    DenseLayerData d_layer4;
    std::vector<Conv2DLayerData> d_layers;
    std::vector<ops::Op*> d1_nodes;
    auto d1_logits = discriminator(g_out, d_layer4, d_layers, d1_nodes, weights_in);
    std::vector<ops::Op*> d2_nodes;
    auto d2_logits = discriminator(x, d_layer4, d_layers, d2_nodes, weights_in);

    dbl_t* logits_0 = new dbl_t[BATCH];
    dbl_t* logits_1 = new dbl_t[BATCH];
    std::fill_n(logits_0, BATCH, 0);
    std::fill_n(logits_1, BATCH, 1);
    auto logits0 = builder.input(ops::Shape({BATCH, 1}));
    auto logits1 = builder.input(ops::Shape({BATCH, 1}));

    auto g_loss = builder.sigmoid_cross_entropy(logits1, d1_logits);
    auto d1_loss = builder.sigmoid_cross_entropy(logits0, d1_logits);
    auto d2_loss = builder.sigmoid_cross_entropy(logits1, d2_logits);
    auto d_loss = builder.add(d1_loss, d2_loss);


    std::vector<ops::Variable*> g_vars;
    for (auto l : g_layers)
    {
        g_vars.push_back(dynamic_cast<ops::Variable*>(l.w));
        g_vars.push_back(dynamic_cast<ops::Variable*>(l.b));
    }
    g_vars.push_back(dynamic_cast<ops::Variable*>(g_layer0.w));
    g_vars.push_back(dynamic_cast<ops::Variable*>(g_layer0.b));
    std::vector<ops::Variable*> d_vars;
    for (auto l : d_layers)
    {
        d_vars.push_back(dynamic_cast<ops::Variable*>(l.w));
        d_vars.push_back(dynamic_cast<ops::Variable*>(l.b));
    }
    d_vars.push_back(dynamic_cast<ops::Variable*>(d_layer4.w));
    d_vars.push_back(dynamic_cast<ops::Variable*>(d_layer4.b));


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
    
    AdamOptimizer g_adam(LEARNING_RATE, BETA1);
    //SGDOptimizer g_adam(0.5);
    auto g_opti = g_adam.minimize(g_loss, g_vars);
    AdamOptimizer d_adam(LEARNING_RATE, BETA1);
    //SGDOptimizer d_adam(0.5);
    auto d_opti = g_adam.minimize(d_loss, d_vars);


    auto input = tocha::Tensors::load("input.npz");
    dbl_t* data_x = reinterpret_cast<dbl_t*>(input.arr()[0].data);
    //dbl_t* data_z = reinterpret_cast<dbl_t*>(input.arr()[1].data);
    

    for (std::size_t i = 0; i < 1000; ++i)
    {
        dbl_t* data_z = new dbl_t[BATCH * Z_DIM];
        NormalInitializer init(0, 1);
        for (std::size_t i = 0; i < BATCH * Z_DIM; ++i)
            data_z[i] = init.next();
    


        generate_samples("sample" + std::to_string(i) + ".jpg", g_out, z, data_z);
        save_weights("weights" + std::to_string(i) + ".tbin", g_vars, d_vars);

        
    
        tocha::Tensors gout;

        gout.add(tocha::Tensor::f32(Z_DIM, 4 * 4 * 512));
        dbl_t* g_dw0_out = reinterpret_cast<dbl_t*>(gout.arr()[0].data);
        gout.add(tocha::Tensor::f32(4 * 4 * 512));
        dbl_t* g_db0_out = reinterpret_cast<dbl_t*>(gout.arr()[1].data);
        gout.add(tocha::Tensor::f32(5, 5, 256, 512));
        dbl_t* g_dw1_out = reinterpret_cast<dbl_t*>(gout.arr()[2].data);
        gout.add(tocha::Tensor::f32(256));
        dbl_t* g_db1_out = reinterpret_cast<dbl_t*>(gout.arr()[3].data);
        gout.add(tocha::Tensor::f32(5, 5, 128, 256));
        dbl_t* g_dw2_out = reinterpret_cast<dbl_t*>(gout.arr()[4].data);
        gout.add(tocha::Tensor::f32(128));
        dbl_t* g_db2_out = reinterpret_cast<dbl_t*>(gout.arr()[5].data);
        gout.add(tocha::Tensor::f32(5, 5, 64, 128));
        dbl_t* g_dw3_out = reinterpret_cast<dbl_t*>(gout.arr()[6].data);
        gout.add(tocha::Tensor::f32(64));
        dbl_t* g_db3_out = reinterpret_cast<dbl_t*>(gout.arr()[7].data);
        gout.add(tocha::Tensor::f32(5, 5, 3, 64));
        dbl_t* g_dw4_out = reinterpret_cast<dbl_t*>(gout.arr()[8].data);
        gout.add(tocha::Tensor::f32(3));
        dbl_t* g_db4_out = reinterpret_cast<dbl_t*>(gout.arr()[9].data);
        gout.add(tocha::Tensor::f32(5, 5, 3, 64));


        dbl_t* d_dw0_out = reinterpret_cast<dbl_t*>(gout.arr()[10].data);
        gout.add(tocha::Tensor::f32(64));
        dbl_t* d_db0_out = reinterpret_cast<dbl_t*>(gout.arr()[11].data);
        gout.add(tocha::Tensor::f32(5, 5, 64, 128));
        dbl_t* d_dw1_out = reinterpret_cast<dbl_t*>(gout.arr()[12].data);
        gout.add(tocha::Tensor::f32(128));
        dbl_t* d_db1_out = reinterpret_cast<dbl_t*>(gout.arr()[13].data);
        gout.add(tocha::Tensor::f32(5, 5, 128, 256));
        dbl_t* d_dw2_out = reinterpret_cast<dbl_t*>(gout.arr()[14].data);
        gout.add(tocha::Tensor::f32(256));
        dbl_t* d_db2_out = reinterpret_cast<dbl_t*>(gout.arr()[15].data);
        gout.add(tocha::Tensor::f32(5, 5, 256, 512));
        dbl_t* d_dw3_out = reinterpret_cast<dbl_t*>(gout.arr()[16].data);
        gout.add(tocha::Tensor::f32(512));
        dbl_t* d_db3_out = reinterpret_cast<dbl_t*>(gout.arr()[17].data);
        gout.add(tocha::Tensor::f32(8192, 1));
        dbl_t* d_dw4_out = reinterpret_cast<dbl_t*>(gout.arr()[18].data);
        gout.add(tocha::Tensor::f32(1));
        dbl_t* d_db4_out = reinterpret_cast<dbl_t*>(gout.arr()[19].data);


        graph.run({g_dw0, g_db0, g_dw1, g_db1, g_dw2, g_db2, g_dw3, g_db3, g_dw4, g_db4,
                   d_dw0, d_db0, d_dw1, d_db1, d_dw2, d_db2, d_dw3, d_db3, d_dw4, d_db4},
            {{x, {data_x, ops::Shape({BATCH, 64, 64, 3})}},
             {z, {data_z, ops::Shape({BATCH, Z_DIM})}},
             {logits0, {logits_0, ops::Shape({BATCH, 1})}},
             {logits1, {logits_1, ops::Shape({BATCH, 1})}},
            },
            {g_dw0_out, g_db0_out, g_dw1_out, g_db1_out, g_dw2_out, g_db2_out,
             g_dw3_out, g_db3_out, g_dw4_out, g_db4_out,
             d_dw0_out, d_db0_out, d_dw1_out, d_db1_out, d_dw2_out, d_db2_out,
             d_dw3_out, d_db3_out, d_dw4_out, d_db4_out
            });
    
        gout.save("grads" + std::to_string(i) + ".tbin");

        graph.run({d_opti},
                  {
                      {logits0, {logits_0, ops::Shape({BATCH, 1})}},
                      {logits1, {logits_1, ops::Shape({BATCH, 1})}},
                      {z, {data_z, ops::Shape({BATCH, Z_DIM})}},
                      {x, {data_x, ops::Shape({BATCH, 64, 64, 3})}}},
                  {nullptr});

        for (std::size_t i = 0; i < 2; ++i)
            graph.run({g_opti},
                      {
                          {logits1, {logits_1, ops::Shape({BATCH, 1})}},
                          {z, {data_z, ops::Shape({BATCH, Z_DIM})}}},
                      {nullptr});

        delete[] data_z;
    }
    
    /*
    if (args.has_option("train"))
    {
        std::vector<std::size_t> idxs_all(DATASET_LEN);
        for (std::size_t i = 0; i < DATASET_LEN; ++i)
            idxs_all[i] = i;
        auto rng = std::default_random_engine {};
        dbl_t* z_batch = new dbl_t[BATCH * Z_DIM];
        NormalInitializer z_init(0, 1);
    
        int nepochs = std::atoi(args.get_option("train").c_str());
        int niters = DATASET_LEN / BATCH;

        for (int i = 0; i < nepochs; ++i)
        {
            std::shuffle(idxs_all.begin(), idxs_all.end(), rng);

            for (int j = 0; j < niters; ++j)
            {
                std::vector<std::size_t> idxs_batch(idxs_all.begin() + j * BATCH, idxs_all.begin() + (j + 1) * BATCH);

                dbl_t* x_batch = celeba::load(idxs_batch);
                for (std::size_t i = 0; i < BATCH * Z_DIM; ++i)
                    z_batch[i] = z_init.next();

                dbl_t d_loss_val;
                dbl_t g_loss_val;

                //update network
                graph.run({d_opti, d_loss},
                {
                              {logits0, {logits_0, ops::Shape({BATCH, 1})}},
                              {logits1, {logits_1, ops::Shape({BATCH, 1})}},
                              {z, {z_batch, ops::Shape({BATCH, Z_DIM})}},
                              {x, {x_batch, ops::Shape({BATCH, 64, 64, 3})}}},
                          {nullptr, &d_loss_val});

                for (std::size_t k = 0; k < 2; ++k)
                graph.run({g_opti, g_loss},
                              {
                                  {logits1, {logits_1, ops::Shape({BATCH, 1})}},
                                  {z, {z_batch, ops::Shape({BATCH, Z_DIM})}}},
                              {nullptr, &g_loss_val});

                
                std::cout << "epoch " << i << " [" << j << "/" << niters << "] "
                          << "d_loss = " << d_loss_val << ", g_loss = " << g_loss_val << std::endl;
                
                
                if ((j + 1) % SAVE_STEP == 0)
                {
                    std::string path = "./samples/train_" + std::to_string(i) + "_" + std::to_string(j) + ".jpg";
                    generate_samples(path, g_out, z);
                    if (args.has_option("model"))
                        graph.save_vars(args.get_option("model"));
                }
                
                delete[] x_batch;
            }
            
        }

        delete[] z_batch;
        
        } */   

    
    delete[] logits_0;
    delete[] logits_1;
}
