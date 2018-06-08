#include "../runtime/node.hh"
#include <iostream>

#include "conv2d_access.hh"
#include "conv2d_mat.hh"
#include "conv2d_naive.hh"
#include "conv2d_shared1.hh"
#include "conv2d_shared2.hh"


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

        /*
        std::cout << "X: " << nx << ", " << hx << ", " << wx << ", " << cx << std::endl;
        std::cout << "K: " << hk << ", " << wk << ", " << cx << ", " << ck << std::endl;
        std::cout << "S: " << sh << ", " << sw << std::endl;
        
        std::cout << "P: " << pad_top << ", " << pad_bot << ", " << pad_left << ", " << pad_right
                  << std::endl;

        std::cout << "Y: " << nx << ", " << hy << ", " << wy << ", " << ck << std::endl;
        */

        //conv2d_fwd_shared2(
        //conv2d_fwd_shared1(
        conv2d_fwd_naive(
        //conv2d_fwd_mat(
            x, k, y,
            nx, hx, wx, cx,
            pad_top, pad_left, pad_bot, pad_right,
            hk, wk, ck,
            hy, wy,
            sh, sw
            );
        
    }
    
}
