#include "conv2d.hh"

#include <algorithm>
#include "../../conv2d/tensor4.hh"
#include "../../conv2d/conv.hh"
#include "../../conv2d/tensor4_access.hh"

namespace cpu
{

    void conv2d_sp(const dbl_t* x, std::size_t nx, std::size_t hx, std::size_t wx, std::size_t cx,
                   const dbl_t* k, std::size_t hk, std::size_t wk, std::size_t ck,
                   dbl_t* y,
                   std::size_t sh, std::size_t sw,
                   std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
    {

        /*
        Tensor4 tx(nx, hx, wx, cx);
        std::copy(x, x + tx.size, tx.data);
        Tensor4 tk(hk, wk, cx, ck);
        std::copy(k, k + tk.size, tk.data);

        Tensor4 ty = ::conv2d_sp(tx, tk, sh, sw, p1, p2, p3, p4);
        std::copy(ty.data, ty.data + ty.size, y);
        */
        acc::conv2d_sp(x, k, y, nx, hx, wx, cx, hk, wk, ck, sh, sw, p1, p2, p3, p4);
    }

    void conv2d_sp_dk(const dbl_t* x, std::size_t nx, std::size_t hx, std::size_t wx, std::size_t cx,
                      const dbl_t* dy, std::size_t hy, std::size_t wy, std::size_t ck,
                      dbl_t* dk,
                      std::size_t sh, std::size_t sw,
                      std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
    {
        /*
        Tensor4 tx(nx, hx, wx, cx);
        std::copy(x, x + tx.size, tx.data);
        Tensor4 tdy(nx, hy, wy, ck);
        std::copy(dy, dy + tdy.size, tdy.data);

        Tensor4 tdk = ::conv2d_sp_dk(tx, tdy, sh, sw, p1, p2, p3, p4);
        std::copy(tdk.data, tdk.data + tdk.size, dk);
        */
        acc::conv2d_sp_dk(x, dy, dk, nx, hx, wx, cx, hy, wy, ck, sh, sw, p1, p2, p3, p4);
    }

    void conv2d_sp_dx(const dbl_t* k, std::size_t hk, std::size_t wk, std::size_t cx, std::size_t ck,
                      const dbl_t* dy, std::size_t nx, std::size_t hy, std::size_t wy,
                      dbl_t* dx,
                      std::size_t sh, std::size_t sw,
                      std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
    {
        /*
        Tensor4 tk(hk, wk, cx, ck);
        std::copy(k, k + tk.size, tk.data);
        Tensor4 tdy(nx, hy, wy, ck);
        std::copy(dy, dy + tdy.size, tdy.data);

        Tensor4 tdx = ::conv2d_sp_dx(tk, tdy, sh, sw, p1, p2, p3, p4);
        std::copy(tdx.data, tdx.data + tdx.size, dx);
        */
        acc::conv2d_sp_dx(k, dy, dx, hk, wk, cx, nx, hy, wy, ck, sh, sw, p1, p2, p3, p4);
    }
    
}
