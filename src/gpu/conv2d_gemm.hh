#pragma once

#include <cmath>

namespace gpu
{

    namespace
    {
      template<int widthA, int widthB, int blockDim, int maxA, int maxB>
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
             if (aIndex < maxA)
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
             if ((bPIndex + indexPartial) < maxB)
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
        const int hIdx = (index - wIdx * chSize * nbFilter - chIdx * nbFilter
                        - filterIdx)/(wSize * chSize * nbFilter);
        const int nIndex = filterIdx * hSize * wSize * chSize
                + chIdx * wSize * hSize + hIdx * wSize + wIdx;
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

    template<int width, int P, int Q, int nbFilter>
    __global__
    void transform_res(const dbl_t *resConv, dbl_t *transf)
    {
      const int blkIdx = blockIdx.y * width * 4 + blockIdx.x * 16;
      const int thIdx = threadIdx.y * width + threadIdx.x;
      const int realIdx = blkIdx + thIdx;
      const int nbFIdx = blockIdx.y * 4 + threadIdx.y;//realIdx / width;
      const int tmp0 = nbFIdx * width;
      const int batchIdx = (realIdx - tmp0) / (P * Q);
      const int tmp1 = batchIdx * P * Q;
      const int tmp2 = realIdx - tmp0 - tmp1;
      const int hIdx = tmp2 / P;
      const int wIdx = tmp2 % P;

      transf[batchIdx * P * Q * nbFilter + hIdx * P * nbFilter
              + wIdx * nbFilter + nbFIdx]
              = resConv[realIdx];
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
      mat_mul_cuda<75, 65536, 16, 4800, 4915200><<<dimGridConv, dimBlockConv>>>(
                      newKernelCuda, newInputCuda, resConvCuda);

      cudaDeviceSynchronize();

      dim3 dimGridTransf(4096, 16);
      dim3 dimBlockTransf(16, 4);
      transform_res<65536, 32, 32, 64><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

      cudaDeviceSynchronize();

      cudaFree(newKernelCuda);
      cudaFree(newInputCuda);
      cudaFree(resConvCuda);
    }

    void conv2d_d1_caller(const float *data, const float *ker, float *res)
    {
      dbl_t *newKernelCuda;
      constexpr int totalKernelSize1 = 5 * 5 * 128 * 64;
      cudaMalloc((void**)&newKernelCuda, sizeof(dbl_t) * totalKernelSize1);
      dim3 dimGrid(totalKernelSize1 / 32);
      dim3 dimBlock(32);
      ker_transform_cuda<5, 5, 64, 128><<<dimGrid, dimBlock>>>(ker, newKernelCuda);

      cudaDeviceSynchronize();

      dbl_t *newInputCuda;
      constexpr int newInputSize1 = 64 * 5 * 5 * 64 * 16 * 16;
      cudaMalloc((void**)&newInputCuda, sizeof(dbl_t) * newInputSize1);

      #pragma unroll
      for (int b = 0; b < 64; ++b)
      {
        im2col<16, 16, 16384, 32, 32, 5, 5, 2, 1, 1, 64, 64>
                <<<16, 1024>>>(newInputCuda, data, b);
      }

      cudaDeviceSynchronize();

      dbl_t *resConvCuda;
      constexpr int resSize1 = 128 * 64 * 16 * 16;
      cudaMalloc((void**)&resConvCuda, sizeof(dbl_t) * resSize1);
      dim3 dimBlockConv(16, 4);
      dim3 dimGridConv(256, 8);
      mat_mul_cuda<1600, 16384, 16, 204800, 26214400><<<dimGridConv, dimBlockConv>>>(
                      newKernelCuda, newInputCuda, resConvCuda);

      cudaDeviceSynchronize();


      dim3 dimGridTransf(1024, 32);
      dim3 dimBlockTransf(16, 4);
      transform_res<16384, 16, 16, 128><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

      cudaDeviceSynchronize();

      cudaFree(newKernelCuda);
      cudaFree(newInputCuda);
      cudaFree(resConvCuda);
    }

    void conv2d_d2_caller(const float *data, const float *ker, float *res)
    {
      dbl_t *newKernelCuda;
      constexpr int totalKernelSize1 = 5 * 5 * 128 * 256;
      cudaMalloc((void**)&newKernelCuda, sizeof(dbl_t) * totalKernelSize1);
      dim3 dimGrid(totalKernelSize1 / 32);
      dim3 dimBlock(32);
      ker_transform_cuda<5, 5, 128, 256><<<dimGrid, dimBlock>>>(ker, newKernelCuda);

      cudaDeviceSynchronize();

      dbl_t *newInputCuda;
      constexpr int newInputSize1 = 5 * 5 * 128 * 64 * 8 * 8;
      cudaMalloc((void**)&newInputCuda, sizeof(dbl_t) * newInputSize1);

      #pragma unroll
      for (int b = 0; b < 64; ++b)
      {
        im2col<8, 8, 8192, 16, 16, 5, 5, 2, 1, 1, 64, 128>
                <<<8, 1024>>>(newInputCuda, data, b);
      }

      cudaDeviceSynchronize();

      dbl_t *resConvCuda;
      constexpr int resSize1 = 256 * 64 * 8 * 8;
      cudaMalloc((void**)&resConvCuda, sizeof(dbl_t) * resSize1);
      dim3 dimBlockConv(16, 4);
      dim3 dimGridConv(64, 16);
      mat_mul_cuda<3200, 4096, 16, 819200, 13107200><<<dimGridConv, dimBlockConv>>>(
                      newKernelCuda, newInputCuda, resConvCuda);

      cudaDeviceSynchronize();


      dim3 dimGridTransf(256, 64);
      dim3 dimBlockTransf(16, 4);
      transform_res<4096, 8, 8, 256><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

      cudaDeviceSynchronize();

      cudaFree(newKernelCuda);
      cudaFree(newInputCuda);
      cudaFree(resConvCuda);
    }

    void conv2d_d3_caller(const float *data, const float *ker, float *res)
    {
      dbl_t *newKernelCuda;
      constexpr int totalKernelSize1 = 5 * 5 * 256 * 512;
      cudaMalloc((void**)&newKernelCuda, sizeof(dbl_t) * totalKernelSize1);
      dim3 dimGrid(totalKernelSize1 / 32);
      dim3 dimBlock(32);
      ker_transform_cuda<5, 5, 256, 512><<<dimGrid, dimBlock>>>(ker, newKernelCuda);

      cudaDeviceSynchronize();

      dbl_t *newInputCuda;
      constexpr int newInputSize1 = 5 * 5 * 256 * 64 * 4 * 4;
      cudaMalloc((void**)&newInputCuda, sizeof(dbl_t) * newInputSize1);

      #pragma unroll
      for (int b = 0; b < 64; ++b)
      {
        im2col<4, 4, 4096, 8, 8, 5, 5, 2, 1, 1, 64, 256>
                <<<4, 1024>>>(newInputCuda, data, b);
      }

      cudaDeviceSynchronize();

      dbl_t *resConvCuda;
      constexpr int resSize1 = 512 * 64 * 4 * 4;
      cudaMalloc((void**)&resConvCuda, sizeof(dbl_t) * resSize1);
      dim3 dimBlockConv(16, 4);
      dim3 dimGridConv(16, 32);
      mat_mul_cuda<6400, 1024, 16, 3276800, 6553600><<<dimGridConv, dimBlockConv>>>(
                      newKernelCuda, newInputCuda, resConvCuda);

      cudaDeviceSynchronize();


      dim3 dimGridTransf(64, 128);
      dim3 dimBlockTransf(16, 4);
      transform_res<1024, 4, 4, 512><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

      cudaDeviceSynchronize();

      cudaFree(newKernelCuda);
      cudaFree(newInputCuda);
      cudaFree(resConvCuda);
    }
  }
}
