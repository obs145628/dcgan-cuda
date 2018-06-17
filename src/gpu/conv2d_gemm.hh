#pragma once

#include <cmath>

namespace gpu
{

    namespace
    {
      template<int widthA, int widthB, int blockDim>
      __global__
      void mat_mul_cuda(const dbl_t *A, const dbl_t *B, dbl_t *C)
      {
         __shared__ dbl_t A_tile[blockDim * blockDim];
         dbl_t cRes[blockDim] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

         const int aStart = widthA * blockDim * blockIdx.y;
         const int aEnd = aStart + widthA - 1;
         const int bStart = blockDim * 4 * blockIdx.x;
         const int bStep = blockDim * widthB;

         int cIdx = widthB * blockDim * blockIdx.y + blockDim * blockIdx.x * 4
                         + blockDim * threadIdx.y + threadIdx.x;

         for (int aIdx = aStart, bIdx = bStart; aIdx <= aEnd;
              aIdx += blockDim, bIdx += bStep)
         {

           #pragma unroll
           for (int i = 0; i < 16; i += 4)
           {
             const int aIndex = aIdx + widthA * (i + threadIdx.y) + threadIdx.x;
             if (aIndex < 4800)
               A_tile[i + threadIdx.y + blockDim * threadIdx.x] = A[aIndex];
           }

           __syncthreads();

           const int bPIndex = bIdx + blockDim * threadIdx.y + threadIdx.x;
           const dbl_t *bPartial = &B[bPIndex];
           int indexPartial = 0;
           int tileIndex = 0;

           #pragma unroll
           for (int i = 0; i < blockDim; ++i)
           {
             if (bPIndex + indexPartial < 4915200)
             {
               const dbl_t bVal = bPartial[indexPartial];

               #pragma unroll
               for (int j = 0; j < 16; ++j)
                 cRes[j] += A_tile[tileIndex + j] * bVal;
               tileIndex += blockDim;
               indexPartial += widthB;
             }
           }

           __syncthreads();
         }

         #pragma unroll
         for (int i = 0; i < blockDim; ++i)
         {
           C[cIdx] = cRes[i];
           cIdx += widthB;
         }
      }

      template<int hSize, int wSize, int chSize, int nbFilter>
      __global__
      void ker_transform_cuda(const dbl_t *ker, dbl_t *res)
      {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int filterIdx = index % nbFilter;
        const int chIdx = ((index - filterIdx) % (chSize * nbFilter))/nbFilter;
        const int wIdx = ((index - filterIdx - (chIdx * nbFilter))
                      % (chSize * nbFilter * wSize))/(chSize * nbFilter);
        const int hIdx = (index - wIdx * chSize * nbFilter - chIdx * nbFilter - filterIdx)/(wSize * chSize * nbFilter);
        const int nIndex = filterIdx * hSize * wSize * chSize + chIdx * wSize * hSize + hIdx * wSize + wIdx;
        res[nIndex] = ker[index];
      }

    template<int P, int Q, int depthSize, int width, int height,
             int widthK, int heightK, int stride, int padL, int padTop,
             int batchSize, int chSize>
    __global__ void
    im2col(float* res,
           float const* data,
           const int batchIdx)
    {
      int index = threadIdx.x + blockIdx.x * blockDim.x ;
      if (index < depthSize)
      {
        const int x = index % P;
        int y = index / P;
        const int z = y / Q;
        y %= Q;

        const int xOldIdx = x * stride - padL;
        const int yOldIdx = y * stride - padTop;
        data += batchIdx * (chSize * width * height)
                + (yOldIdx * chSize * width)
                + xOldIdx * chSize + z;

        const int resOffset = (P*Q*batchSize*widthK*heightK) * z
                + batchIdx * (P*Q)
                + y * P + x;
        res += resOffset;

        #pragma unroll
        for (int dj = 0; dj < heightK; ++dj)
        {
          #pragma unroll
          for (int di = 0; di < widthK; ++di)
          {
            if (yOldIdx + dj >= 0 && yOldIdx + dj < height
                && xOldIdx + di >= 0 &&  xOldIdx + di < width)
              *res = data[dj * chSize * width + di * chSize];
            else
              *res = 0;
            res += (P*Q*batchSize);
          }
        }
      }
    }

    __global__
    void transform_res(const dbl_t *resConv, dbl_t *transf)
    {
      int blkIdx = blockIdx.y * 65536 * 4 + blockIdx.x * 16;
      const int thIdx = threadIdx.y * 65536 + threadIdx.x;
      const int realIdx = blkIdx + thIdx;
      const int nbFIdx = realIdx / 65536;
      const int tmp0 = nbFIdx * 65536;
      const int batchIdx = (realIdx - tmp0) / 1024;
      const int tmp1 = batchIdx * 1024;
      const int tmp2 = realIdx - tmp0 - tmp1;
      const int hIdx = tmp2 / 32;
      const int wIdx = tmp2 % 32;

      transf[batchIdx * 65536 + hIdx * 32 * 64 + wIdx * 64 + nbFIdx] = resConv[realIdx];
    }

    void conv2d_d0_caller(const float *data, const float *ker, float *res)
    {
      dbl_t *newKernelCuda;
      constexpr int totalKernelSize = 5 * 5 * 3 * 64;
      cudaMalloc((void**)&newKernelCuda, sizeof(dbl_t) * totalKernelSize);
      dim3 dimGrid(totalKernelSize / 32);
      dim3 dimBlock(32);
      ker_transform_cuda<5, 5, 3, 64><<<dimGrid, dimBlock>>>(ker, newKernelCuda);

      cudaDeviceSynchronize();

      dbl_t *newInputCuda;
      constexpr int newInputSize = 3 * 5 * 5 * 64 * 32 * 32;
      cudaMalloc((void**)&newInputCuda, sizeof(dbl_t) * newInputSize);

      #pragma unroll
      for (int b = 0; b < 64; ++b)
      {
        im2col<32, 32, 3072, 64, 64, 5, 5, 2, 1, 1, 64, 3>
                <<<3, 1024>>>(newInputCuda, data, b);
      }

      cudaDeviceSynchronize();

      dbl_t *resConvCuda;
      constexpr int resSize = 64 * 64 * 32 * 32;
      cudaMalloc((void**)&resConvCuda, sizeof(dbl_t) * resSize);
      dim3 dimBlockConv(16, 4);
      dim3 dimGridConv(1024, 4);
      mat_mul_cuda<75, 65536, 16><<<dimGridConv, dimBlockConv>>>(
                      newKernelCuda, newInputCuda, resConvCuda);

      cudaDeviceSynchronize();

      dim3 dimGridTransf(4096, 16);
      dim3 dimBlockTransf(16, 4);
      transform_res<<<dimGridTransf, dimBlockTransf>>>(resConvCuda, res);

      cudaDeviceSynchronize();

      cudaFree(newKernelCuda);
      cudaFree(newInputCuda);
      cudaFree(resConvCuda);
    }
  }
}
