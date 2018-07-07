#include "../runtime/node.hh"
#include <iostream>

#include "conv2d_access.hh"
#include "conv2d_mat.hh"
#include "conv2d_naive.hh"
#include "conv2d_shared1.hh"
#include "conv2d_shared2.hh"
#include "conv2d_gemm.hh"

namespace gpu
{

    void kernel_conv2d(rt::Node* node)
    {

        const dbl_t* x = node->in1;
        std::size_t nx = node->sizes1[0];
        std::size_t hx = node->sizes1[1];
        std::size_t wx = node->sizes1[2];
        std::size_t cx = node->sizes1[3];

        const dbl_t* k = node->in2;
        std::size_t hk = node->sizes2[0];
        std::size_t wk = node->sizes2[1];
        std::size_t ck = node->sizes2[3];

        dbl_t* y = node->out1;
        std::size_t sh = node->intconst[0];
        std::size_t sw = node->intconst[1];
        int pad_height = node->int_cons1;
        int pad_width = node->int_cons2;

        std::size_t pad_top = pad_height / 2;
        std::size_t pad_left = pad_width / 2;
        std::size_t pad_bot = pad_height - pad_top;
        std::size_t pad_right = pad_width - pad_left;

        std::size_t hy = (hx + pad_top + pad_bot - hk) / sh + 1;
        std::size_t wy = (wx + pad_left + pad_right - wk) / sw + 1;


        if (nx == 64 && hx == 64 && wx == 64 && cx == 3)
          conv2d_d0_caller(x, k, y);
        else if (nx == 64 && hx == 32 && wx == 32 && cx == 64)
          conv2d_d1_caller(x, k, y);
        else if (nx == 64 && hx == 16 && wx == 16 && cx == 128)
          conv2d_d2_caller(x, k, y);
        else if (nx == 64 && hx == 8 && wx == 8 && cx == 256)
          conv2d_d3_caller(x, k, y);
        else
          conv2d_fwd_shared2(
          //conv2d_fwd_shared1(
          //conv2d_fwd_naive(
          //conv2d_fwd_mat(
            x, k, y,
            nx, hx, wx, cx,
            pad_top, pad_left, pad_bot, pad_right,
            hk, wk, ck,
            hy, wy,
            sh, sw
            );
    }

    void conv2d_sp_dx(const dbl_t* k, std::size_t hk, std::size_t wk, std::size_t cx, std::size_t ck,
                      const dbl_t* dy, std::size_t nx, std::size_t hy, std::size_t wy,
                      dbl_t* dx,
                      std::size_t sh, std::size_t sw,
                      std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);

    void kernel_conv2d_input_grad(rt::Node* node)
    {

        const dbl_t* k = node->in2;
        std::size_t hk = node->sizes2[0];
        std::size_t wk = node->sizes2[1];
        std::size_t cx = node->sizes2[2];
        std::size_t ck = node->sizes2[3];

        const dbl_t* dy = node->in1;
        std::size_t nx = node->sizes1[0];
        std::size_t hy = node->sizes1[1];
        std::size_t wy = node->sizes1[2];

        dbl_t* dx = node->out1;
        std::size_t sh = node->intconst[0];
        std::size_t sw = node->intconst[0];

        std::size_t pad_height = (hy - 1) * sh - node->intconst2[0] + hk;
        std::size_t pad_width = (wy - 1) * sw - node->intconst2[1] + wk;
        std::size_t pad_top = pad_height / 2;
        std::size_t pad_left = pad_width / 2;
        std::size_t pad_bot = pad_height - pad_top;
        std::size_t pad_right = pad_width - pad_left;


        std::size_t hx = sh * (hy - 1) + hk - pad_height;
        std::size_t wx = sw * (wy - 1) + wk - pad_width;

        if (nx == 64 && hy == 32 && wy == 32 && ck == 64)
            conv2d_d0_dx_caller(dy, k, dx);
        else if (nx == 64 && hy == 16 && wy == 16 && ck == 128)
            conv2d_d1_dx_caller(dy, k, dx);
        else if (nx == 64 && hy == 8 && wy == 8 && ck == 256)
            conv2d_d2_dx_caller(dy, k, dx);
        else if (nx == 64 && hy == 4 && wy == 4 && ck == 512)
            conv2d_d3_dx_caller(dy, k, dx);
        else
          conv2d_dx_naive(
            k, dy, dx,
            nx, hx, wx, cx,
            pad_top, pad_left, pad_bot, pad_right,
            hk, wk, ck,
            hy, wy,
            sh, sw
            );
        
    }

    void kernel_conv2d_kernel_grad(rt::Node* node)
    {


        const dbl_t* x = node->in2;
        std::size_t nx = node->sizes2[0];
        std::size_t hx = node->sizes2[1];
        std::size_t wx = node->sizes2[2];
        std::size_t cx = node->sizes2[3];

        const dbl_t* dy = node->in1;
        std::size_t hy = node->sizes1[1];
        std::size_t wy = node->sizes1[2];
        std::size_t cy = node->sizes1[3];

        dbl_t* dk = node->out1;
        std::size_t sh = node->intconst[0];
        std::size_t sw = node->intconst[1];

        std::size_t pad_height = node->intconst2[0];
        std::size_t pad_width = node->intconst2[1];
        std::size_t pad_top = pad_height / 2;
        std::size_t pad_left = pad_width / 2;
        std::size_t pad_bot = pad_height - pad_top;
        std::size_t pad_right = pad_width - pad_left;

        std::size_t hk = hx + pad_height - sh * (hy - 1);
        std::size_t wk = wx + pad_width - sw * (wy - 1);

        if (nx == 64 && wy == 32 && hy == 32 && cy == 64)
          conv2d_d0_dk_caller(x, dy, dk);
        else if (nx == 64 && wy == 16 && hy == 16 && cy == 128)
          conv2d_d1_dk_caller(x, dy, dk);
        else if (nx == 64 && wy == 8 && hy == 8 && cy == 256)
          conv2d_d2_dk_caller(x, dy, dk);
        else if (nx == 64 && wy == 4 && hy == 4 && cy == 512)
          conv2d_d3_dk_caller(x, dy, dk);
        else
          conv2d_dk_naive(
            x, dy, dk,
            nx, hx, wx, cx,
            pad_top, pad_left, pad_bot, pad_right,
            hk, wk, cy,
            hy, wy,
            sh, sw
            );
    }

    void kernel_conv2d_transpose(rt::Node* node)
    {

        const dbl_t* x = node->in1;
        std::size_t nx = node->sizes2[0];
        std::size_t hx = node->sizes2[1];
        std::size_t wx = node->sizes2[2];
        std::size_t cx = node->sizes2[3];

        const dbl_t* k = node->in2;
        std::size_t hk = node->sizes3[0];
        std::size_t wk = node->sizes3[1];
        std::size_t ck = node->sizes3[2];

        dbl_t* y = node->out1;
        std::size_t sh = node->intconst[0];
        std::size_t sw = node->intconst[1];

        std::size_t pad_height = (hx - 1) * sh - node->sizes1[1] + hk;
        std::size_t pad_width = (wx - 1) * sw - node->sizes1[2] + wk;
        std::size_t pad_top = pad_height / 2;
        std::size_t pad_left = pad_width / 2;
        std::size_t pad_bot = pad_height - pad_top;
        std::size_t pad_right = pad_width - pad_left;

        std::size_t hy = sh * (hx - 1) + hk - pad_height;
        std::size_t wy = sw * (wx - 1) + wk - pad_width;

        if (nx == 64 && hx == 32 && wx == 32 && cx == 64)
            conv2d_d0_dx_caller(x, k, y);
        else if (nx == 64 && hx == 16 && wx == 16 && cx == 128)
            conv2d_d1_dx_caller(x, k, y);
        else if (nx == 64 && hx == 8 && wx == 8 && cx == 256)
            conv2d_d2_dx_caller(x, k, y);
        else if (nx == 64 && hx == 4 && wx == 4 && cx == 512)
            conv2d_d3_dx_caller(x, k, y);
        else
            conv2d_dx_naive(
                k, x, y,
                nx, hy, wy, ck,
                pad_top, pad_left, pad_bot, pad_right,
                hk, wk, cx,
                hx, wx,
                sh, sw
                );
    }

    void kernel_conv2d_transpose_input_grad(rt::Node* node)
    {

        const dbl_t* k = node->in2;
        std::size_t hk = node->sizes2[0];
        std::size_t wk = node->sizes2[1];
        std::size_t ck = node->sizes2[2];
        std::size_t cx = node->sizes2[3];

        const dbl_t* dy = node->in1;
        std::size_t nx = node->sizes1[0];
        std::size_t hy = node->sizes1[1];
        std::size_t wy = node->sizes1[2];

        dbl_t* dx = node->out1;
        std::size_t sh = node->intconst[0];
        std::size_t sw = node->intconst[0];

        std::size_t hx = node->intconst2[0];
        std::size_t wx = node->intconst2[1];

        std::size_t pad_height = (hx - 1) * sh + hk - hy;
        std::size_t pad_width = (wx - 1) * sw + wk - wy;
        std::size_t pad_top = pad_height / 2;
        std::size_t pad_left = pad_width / 2;
        std::size_t pad_bot = pad_height - pad_top;
        std::size_t pad_right = pad_width - pad_left;

        if (nx == 64 && hy == 64 && wy == 64 && ck == 3)
          conv2d_d0_caller(dy, k, dx);
        else if (nx == 64 && hy == 32 && wy == 32 && ck == 64)
          conv2d_d1_caller(dy, k, dx);
        else if (nx == 64 && hy == 16 && wy == 16 && ck == 128)
          conv2d_d2_caller(dy, k, dx);
        else if (nx == 64 && hy == 8 && wy == 8 && ck == 256)
          conv2d_d3_caller(dy, k, dx);
        else
            conv2d_fwd_naive(
                dy, k, dx,
                nx, hy, wy, ck,
                pad_top, pad_left, pad_bot, pad_right,
                hk, wk, cx,
                hx, wx,
                sh, sw
                );
    }

    void conv2d_transpose_kernel_grad(const dbl_t* dX1, const dbl_t* X0, const int stride,
                                      const int* dX1_size, const int* X0_size, dbl_t* out);

    void kernel_conv2d_transpose_kernel_grad(rt::Node* node)
    {
        const dbl_t* x = node->in2;
        std::size_t nx = node->sizes2[0];
        std::size_t hx = node->sizes2[1];
        std::size_t wx = node->sizes2[2];
        std::size_t cx = node->sizes2[3];

        const dbl_t* dy = node->in1;
        std::size_t hy = node->sizes1[1];
        std::size_t wy = node->sizes1[2];
        std::size_t cy = node->sizes1[3];

        dbl_t* dk = node->out1;
        std::size_t sh = node->intconst[0];
        std::size_t sw = node->intconst[1];
        std::size_t hk = node->sizes3[0];
        std::size_t wk = node->sizes3[1];

        std::size_t pad_height = (hx - 1) * sh + hk - hy;
        std::size_t pad_width = (wx - 1) * sw + wk - wy;
        std::size_t pad_top = pad_height / 2;
        std::size_t pad_left = pad_width / 2;
        std::size_t pad_bot = pad_height - pad_top;
        std::size_t pad_right = pad_width - pad_left;

        if (nx == 64 && wx == 32 && hx == 32 && cx == 64)
          conv2d_d0_dk_caller(dy, x, dk);
        else if (nx == 64 && wx == 16 && hx == 16 && cx == 128)
          conv2d_d1_dk_caller(dy, x, dk);
        else if (nx == 64 && wx == 8 && hx == 8 && cx == 256)
          conv2d_d2_dk_caller(dy, x, dk);
        else if (nx == 64 && wx == 4 && hx == 4 && cx == 512)
          conv2d_d3_dk_caller(dy, x, dk);        
        else
            conv2d_dk_naive(
                dy, x, dk,
                nx, hy, wy, cy,
                pad_top, pad_left, pad_bot, pad_right,
                hk, wk, cx,
                hx, wx,
                sh, sw
                );
    }

}
