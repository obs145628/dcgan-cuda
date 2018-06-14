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


        /*
        std::cout << "K: " << hk << ", " << wk << ", " << cx << ", " << ck << std::endl;
        std::cout << "Y: " << nx << ", " << hy << ", " << wy << ", " << ck << std::endl;
        std::cout << "X: " << nx << ", " << hx << ", " << wx << ", " << cx << std::endl;
          
        std::cout << "S: " << sh << ", " << sw << std::endl;
        std::cout << "P: " << pad_top << ", " << pad_bot << ", " << pad_left << ", " << pad_right
                  << std::endl;
        */

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

        /*
        std::cout << "K: " << hk << ", " << wk << ", " << cx << ", " << cy << std::endl;
        std::cout << "Y: " << nx << ", " << hy << ", " << wy << ", " << cy << std::endl;
        std::cout << "X: " << nx << ", " << hx << ", " << wx << ", " << cx << std::endl;
          
        std::cout << "S: " << sh << ", " << sw << std::endl;
        std::cout << "P: " << pad_top << ", " << pad_bot << ", " << pad_left << ", " << pad_right
                  << std::endl;
        */

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

        /*
        std::cout << "X: " << nx << ", " << hx << ", " << wx << ", " << cx << std::endl;
        std::cout << "K: " << hk << ", " << wk << ", " << ck << ", " << cx << std::endl;
        std::cout << "Y: " << nx << ", " << hy << ", " << wy << ", " << ck << std::endl;
        std::cout << "S: " << sh << ", " << sw << std::endl;
        std::cout << "P: " << pad_top << ", " << pad_bot << ", " << pad_left << ", " << pad_right
                  << std::endl;
        */


        conv2d_dx_naive(
            k, x, y,
            nx, hy, wy, ck,
            pad_top, pad_left, pad_bot, pad_right,
            hk, wk, cx,
            hx, wx,
            sh, sw
            );
    }

    /*
    inline void conv2d_transpose_input_grad(const dbl_t* dX1, const dbl_t* W1, const int stride,
                                            const int* dX1_size,
                                            const int* W1_size, dbl_t* out, const int* input_size);
    */
    
    void kernel_conv2d_transpose_input_grad(rt::Node* node)
    {

        /*
        conv2d_transpose_input_grad(node->in1, node->in2, node->intconst[0],
                                    node->sizes1,
                                    node->sizes2, node->out1, node->intconst2);
        */
        
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


        /*
        std::cout << "X: " << nx << ", " << hx << ", " << wx << ", " << cx << std::endl;
        std::cout << "K: " << hk << ", " << wk << ", " << ck << ", " << cx << std::endl;
        std::cout << "Y: " << nx << ", " << hy << ", " << wy << ", " << ck << std::endl;  
        std::cout << "S: " << sh << ", " << sw << std::endl;
        std::cout << "P: " << pad_top << ", " << pad_bot << ", " << pad_left << ", " << pad_right
                  << std::endl;
        */

        

        conv2d_fwd_naive(
            dy, k, dx,
            nx, hy, wy, ck,
            pad_top, pad_left, pad_bot, pad_right,
            hk, wk, cx,
            hx, wx,
            sh, sw
            );
    }
    
}
