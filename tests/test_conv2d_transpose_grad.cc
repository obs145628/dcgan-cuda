#include <iostream>
#include <cmath>

#include "../src/memory/types.hh"
#include "../src/ops/vect-sigmoid.hh"
#include "../src/ops/softmax.hh"
#include "../src/ops/variable.hh"
#include "../src/ops/input.hh"
#include "../src/ops/ops-builder.hh"
#include "../src/ops/conv2d-transpose.hh"
#include "../src/ops/reshape.hh"
#include "../src/ops/graph.hh"
#include "../src/ops/mse.hh"
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

    dbl_t input[96] = {
        5.0, 0.0, 4.0,
        -1.0, -1.0, 8.0,
        2.0, -2.0, 9.0,
        3.0, 4.0, -5.0,
        3.0, 3.0, 1.0,
        6.0, 2.0, -5.0,
        3.0, 4.0, -2.0,
        4.0, 0.0, 10.0,
        1.0, 0.0, 5.0,
        0.0, -2.0, 2.0,
        0.0, 1.0, -7.0,
        2.0, 0.0, -4.0,
        3.0, 6.0, -1.0,
        -2.0, 7.0, 0.0,
        -2.0, 1.0, 4.0,
        1.0, 0.0, 9.0,
        8.0, 2.0, -4.0,
        0.0, 1.0, -1.0,
        4.0, -2.0, -8.0,
        1.0, -1.0, 2.0,
        3.0, 0.0, 4.0,
        4.0, 2.0, 9.0,
        3.0, 0.0, -12.0,
        -3.0, 0.0, 4.0,
        1.0, 0.0, 5.0,
        -8.0, 7.0, 0.0,
        0.0, 1.0, 4.0,
        0.0, -9.0, 1.0,
        3.0, 0.0, 4.0,
        2.0, -7.0, 0.0,
        -2.0, 1.0, 3.0,
        -1.0, 0.0, 1.0
    };//2 * 4 * 4 * 3

    dbl_t kernel[24] = {
      1.0, -1.0, 0.0, 1.0,
      1.0, 0.0, 0.0, -1.0,
      1.0, -1.0, 1.0, 0.0,
      0.0, 0.0, 1.0, 1.0,
      0.0, -1.0, 1.0, -1.0,
      1.0, -1.0, 0.0, -1.0
    };//2 * 2 * 2 * 3

    dbl_t y[] = {
 	  4., 7., 2.,-5.,
   	1.,  2., -9., 3.,
	  4., 2., -11., 4.,
	  1., 7.,-5., 1.,
    4.4,   1.,
    -9.,  -9.,
    8.,  -9.,
    8.5,  7.,
    -9.,  -7.,
   -13., 11.,
   5.,   10.,
   -6.,   2.,
    0.,   2.,
   -2.,   1.,
    4.,   -8.,
   0. , -4.,
   -1.,   -7.,
   -6.,   -1.,
    3.,   0.,
   5. , -2.,
    -1.,   2.,
    -1.,  -4.,
   -5. , -11.,
   1.  ,1.,
   2.  , 5.,
   -3. , -1.,
   13. , -6.,
   11. ,-1.,
    1. ,  1.,
    -5.,  -1.,
   -2. , -2.,
    -1.,  -2.,
   1.  , 3.,
   8.  , 2.,
    1. ,  -2.,
   -2. , -2.,
    5. , -4.,
    3. , -6.,
    2. , 2.,
    5. , 2.,
   -7. ,  7.,
   8.  , -7.,
   4.  , -6.,
   -7. ,  8.,
   -3. ,  9.,
   7.  , 3.,
   9.  , 0.,
   0.  , 0.,
   -3. , -1.,
    0. ,  3.,
    -1.,   10.,
    -9.,  -2.,
   -1. ,  4.,
   -3. , -2.,
    2. , 2.,
   9.  , 2.,
   - 4.,  6.,
    -1.,  2.,
    -9.,  -4.,
   11. ,-1.,
   6.,  12.,
   6.,  -6.,
   -1.,   -1.,
   2.,   0.,
    -6.,   2.,
   -0.,  -6.,
    2.,   0.,
    -3.,  2.,
    3.,   3.,
    4.,  -3.,
    2.,   6.,
    7.,  -2.,
    3.,   3.,
  -12.,  -3.,
   -3.,  -3.,
    4.,   3.,
   -4.,  12.,
    2.,  -4.,
   -1.,   1.,
   -2.,   1.,
   -8.,  12.,
   -2.,   4.,
    2.,  -1.,
    4.,  -3.,
    4.,  -1.,
    7.,  -7.,
    9.,  -5.,
   11., -13.,
  -12.,  15.,
   -9.,   9.,
    4.,  -7.,
    1.,  -1.,
    3.,   3.,
    4.,  -3.,
    9.,  -5.,
    7.,  -9.,
   -3.,  -1.,
    2.,   3.,
   -1.,  -1.,
    1.,   1.,
    4.,  -1.,
    7.,  -7.,
    0.,   2.,
    9.,  -2.,
    3.,  -5.,
    0.,  -1.,
    1.,  -2.,
    0.,   0.,
    1.,   1.,
    5.,  -1.,
  -15.,  -1.,
   -7.,  15.,
   -1.,   1.,
    3.,   1.,
    9.,  -9.,
   10.,  -9.,
    5.,  -4.,
    6.,  -6.,
    0.,  -8.,
  -15.,   8.,
    4.,  -4.,
    3.,  -4.,
    1.,  -1.,
   10.,  -1.
    };// 2 * 2 * 2 * 2

    auto& graph = ops::Graph::instance();
    graph.debug_set(true);
    auto& builder = ops::OpsBuilder::instance();

    auto x_node = builder.input(ops::Shape({2, 4, 4, 3}));
    auto k_node = builder.input(ops::Shape({2, 2, 2, 3}));
    auto y_node = builder.input(ops::Shape({2, 8, 8, 2}));
    int strides[] = {2, 2};
    int out_size[] = {2, 8, 8, 2};
    auto y_hat_node = builder.conv2d_transpose(x_node, k_node, out_size, strides);
    auto y_node_reshape = builder.reshape(y_node, ops::Shape({2, 128}));
    auto y_hat_node_reshape = builder.reshape(y_hat_node, ops::Shape({2, 128}));
    auto loss_node_shape = builder.mse(y_node_reshape, y_hat_node_reshape);

    auto dy_hat_node = graph.gradient(loss_node_shape, y_hat_node_reshape);
    auto dx_node = graph.gradient(loss_node_shape, x_node);
    auto dk_node = graph.gradient(loss_node_shape, k_node);

    tocha::Tensors out;
    out.add(tocha::Tensor::f32(2, 4, 4, 3));
    dbl_t* dx = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    out.add(tocha::Tensor::f32(2, 2, 2, 3));
    dbl_t* dk = reinterpret_cast<dbl_t*>(out.arr()[1].data);
    out.add(tocha::Tensor::f32(2, 8, 8, 2));
    dbl_t* dy_hat = reinterpret_cast<dbl_t*>(out.arr()[2].data);

    graph.run({dx_node, dk_node, dy_hat_node},
              {{x_node, {input, ops::Shape({2, 4, 4, 3})}},
               {k_node, {kernel, ops::Shape({2, 2, 2, 3})}},
               {y_node, {y, ops::Shape({2, 8, 8, 2})}}},
	      {dx, dk, dy_hat});

    out.save(argv[1]);
}
