#pragma once

#include <cstddef>
#include "../memory/types.hh"

namespace cpu
{


    /**
     * @param x - 4D tensor (nx, hx, wx, cx)
     * @param nx - number of inputs
     * @param hx - height of each input
     * @param wx - width of each input
     * @param cx - number of channels of each input
     * @param k - 4D tensor (hk, wk, cx, ck)
     * @param hk - height of the kernel
     * @param kw - width of the kernel
     * @param ck - number of filters
     * @param y - 4D tensor (nx, hy, wy, ck)
     * @param sh - size of vertical strides
     * @param sw - size of horizontal strides
     * @param p1 - top padding
     * @param p2 - bottom padding
     * @param p3 - left padding
     * @param p4 - right padding
     *
     * Compute conv2d(x, k) and stores the result into y
     * hy = (hx - hk + p1 + p2) / sh + 1 
     * wy = (wx - wk + p3 + p4) / sw + 1
     */
    void conv2d_sp(const dbl_t* x, std::size_t nx, std::size_t hx, std::size_t wx, std::size_t cx,
                   const dbl_t* k, std::size_t hk, std::size_t wk, std::size_t ck,
                   dbl_t* y,
                   std::size_t sh, std::size_t sw,
                   std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);

    /*
     * @param x - 4D tensor (nx, hx, wx, cx)
     * @param nx - number of inputs
     * @param hx - height of each input
     * @param wx - width of each input
     * @param cx - number of channels of each input
     * @param dy - 4D tensor (nx, hy, wy, ck)
     * @param hy - height of the output
     * @param ky - width of the output
     * @param ck - number of filters
     * @param dy - 4D tensor (nx, dk, hk, ck)
     * @param sh - size of vertical strides
     * @param sw - size of horizontal strides
     * @param p1 - top padding
     * @param p2 - bottom padding
     * @param p3 - left padding
     * @param p4 - right padding
     *
     * Compute dE/dK and store result into dk
     * hy = (hx - hk + p1 + p2) / sh + 1 
     * wy = (wx - wk + p3 + p4) / sw + 1
     */
    void conv2d_sp_dk(const dbl_t* x, std::size_t nx, std::size_t hx, std::size_t wx, std::size_t cx,
                      const dbl_t* dy, std::size_t hy, std::size_t wy, std::size_t ck,
                      dbl_t* dk,
                      std::size_t sh, std::size_t sw,
                      std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);


    /*
     * @param k - 4D tensor (hk, wk, cx, ck)
     * @param hk - height of the kernel
     * @param kw - width of the kernel
     * @param cx - number of channels
     * @param ck - number of filters
     * @param dy - 4D tensor (nx, hy, wy, ck)
     * @param nx - number of images
     * @param hy - height of output
     * @param wy - width of output
     * @param sh - size of vertical strides
     * @param sw - size of horizontal strides
     * @param p1 - top padding
     * @param p2 - bottom padding
     * @param p3 - left padding
     * @param p4 - right padding
     *
     * Compute dE/dx and store result into dx
     * hy = (hx - hk + p1 + p2) / sh + 1 
     * wy = (wx - wk + p3 + p4) / sw + 1
     */    
    void conv2d_sp_dx(const dbl_t* k, std::size_t hk, std::size_t wk, std::size_t cx, std::size_t ck,
                      const dbl_t* dy, std::size_t nx, std::size_t hy, std::size_t wy,
                      dbl_t* dx,
                      std::size_t sh, std::size_t sw,
                      std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);

    
}
