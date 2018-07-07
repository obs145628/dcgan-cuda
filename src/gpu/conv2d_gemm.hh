#pragma once

#include <cmath>
#include <fstream>
#include "../memory/alloc.hh"

namespace gpu
{

    namespace
    {
      #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
      inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
      {
        if (code != cudaSuccess)
        {
          fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
          if (abort) exit(code);
        }
      }

      template<int widthA, int widthB, int blockDim, int maxA, int maxB>
      __global__
      void mat_mul_cuda(const dbl_t *A, const dbl_t *B, dbl_t *C)
      {}

      template<int widthA, int widthB, int maxA, int maxB>
      __global__
      void mat_mul_cuda32(const dbl_t *A, const dbl_t *B, dbl_t *C)
      {
         __shared__ dbl_t A_tile[32 * 32];
         dbl_t cRes[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

         const int aStart = widthA * 32 * blockIdx.y;
         const int aEnd = aStart + widthA - 1;
         const int bStart = 32 * 4 * blockIdx.x;
         const int bStep = 32 * widthB;

         int cIdx = widthB * 32 * blockIdx.y + 32 * blockIdx.x * 4
                         + 32 * threadIdx.y + threadIdx.x;

         for (int aIdx = aStart, bIdx = bStart; aIdx <= aEnd;
              aIdx += 32, bIdx += bStep)
         {

           #pragma unroll
           for (int i = 0; i < 32; i += 4)
           {
              const int aIndex = aIdx + widthA * (i + threadIdx.y) + threadIdx.x;
              if (aIndex < maxA)
                A_tile[i + threadIdx.y + 32 * threadIdx.x] = A[aIndex];
           }

           __syncthreads();

           const int bPIndex = bIdx + 32 * threadIdx.y + threadIdx.x;
           const dbl_t *bPartial = &B[bPIndex];
           int indexPartial = 0;
           int tileIndex = 0;
           const int maxPartial = maxB - bPIndex;

           #pragma unroll
           for (int i = 0; i < 32; ++i)
           {
             if (indexPartial < maxPartial)
             {
               const dbl_t bVal = bPartial[indexPartial];

               #pragma unroll
               for (int j = 0; j < 32; ++j)
                 cRes[j] += A_tile[tileIndex + j] * bVal;
               tileIndex += 32;
               indexPartial += widthB;
             }
           }

           __syncthreads();
         }

         #pragma unroll
         for (int i = 0; i < 32; ++i)
         {
           C[cIdx] = cRes[i];
           cIdx += widthB;
         }
      }

      template<int widthA, int widthB, int maxA, int maxB>
      __global__
      void mat_mul_cuda16(const dbl_t *A, const dbl_t *B, dbl_t *C)
      {
         __shared__ dbl_t A_tile[16 * 16];
         dbl_t cRes[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

         const int aStart = widthA * 16 * blockIdx.y;
         const int aEnd = aStart + widthA - 1;
         const int bStart = 16 * 4 * blockIdx.x;
         const int bStep = 16 * widthB;

         int cIdx = widthB * 16 * blockIdx.y + 16 * blockIdx.x * 4
                         + 16 * threadIdx.y + threadIdx.x;

         for (int aIdx = aStart, bIdx = bStart; aIdx <= aEnd;
              aIdx += 16, bIdx += bStep)
         {

           #pragma unroll
           for (int i = 0; i < 16; i += 4)
           {
             const int aIndex = aIdx + widthA * (i + threadIdx.y) + threadIdx.x;
             if (aIndex < maxA)
               A_tile[i + threadIdx.y + 16 * threadIdx.x] = A[aIndex];
           }

           __syncthreads();

           const int bPIndex = bIdx + 16 * threadIdx.y + threadIdx.x;
           const dbl_t *bPartial = &B[bPIndex];
           int indexPartial = 0;
           int tileIndex = 0;
           const int maxPartial = maxB - bPIndex;

           #pragma unroll
           for (int i = 0; i < 16; ++i)
           {
             if (indexPartial < maxPartial)
             {
               const dbl_t bVal = bPartial[indexPartial];

               #pragma unroll
               for (int j = 0; j < 16; ++j)
                 cRes[j] += A_tile[tileIndex + j] * bVal;
               tileIndex += 16;
               indexPartial += widthB;
             }
           }

           __syncthreads();
         }

         #pragma unroll
         for (int i = 0; i < 16; ++i)
         {
           C[cIdx] = cRes[i];
           cIdx += widthB;
         }
      }

      template<int widthA, int widthB, int maxA, int maxB, int maxC>
      __global__
      void back_mat_mul_cuda16(const dbl_t *A, const dbl_t *B, dbl_t *C)
      {
         __shared__ dbl_t A_tile[16 * 16];
         dbl_t cRes[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

         const int aStart = widthA * 16 * blockIdx.y;
         const int aEnd = aStart + widthA - 1;
         const int bStart = 16 * 4 * blockIdx.x;
         const int bStep = 16 * widthB;

         int cIdx = widthB * 16 * blockIdx.y + 16 * blockIdx.x * 4
                         + 16 * threadIdx.y + threadIdx.x;

         if (cIdx >= maxC)
          return;

         for (int aIdx = aStart, bIdx = bStart; aIdx <= aEnd;
              aIdx += 16, bIdx += bStep)
         {

           #pragma unroll
           for (int i = 0; i < 16; i += 4)
           {
             const int aIndex = aIdx + widthA * (i + threadIdx.y) + threadIdx.x;
             if (aIndex < maxA)
               A_tile[i + threadIdx.y + 16 * threadIdx.x] = A[aIndex];
           }

           __syncthreads();

           const int bPIndex = bIdx + 16 * threadIdx.y + threadIdx.x;
           const dbl_t *bPartial = &B[bPIndex];
           int indexPartial = 0;
           int tileIndex = 0;
           const int maxPartial = maxB - bPIndex;

           #pragma unroll
           for (int i = 0; i < 16; ++i)
           {
             if (indexPartial < maxPartial)
             {
               const dbl_t bVal = bPartial[indexPartial];

               #pragma unroll
               for (int j = 0; j < 16; ++j)
                 cRes[j] += A_tile[tileIndex + j] * bVal;
               tileIndex += 16;
               indexPartial += widthB;
             }
           }

           __syncthreads();
         }

         #pragma unroll
         for (int i = 0; i < 16; ++i)
         {
           if (cIdx < maxC)
             C[cIdx] = cRes[i];
           cIdx += widthB;
         }
      }

      template<int widthA, int widthB, int maxA, int maxB, int maxC>
      __global__
      void back_mat_mul_cuda75(const dbl_t *A, const dbl_t *B, dbl_t *C)
      {
         __shared__ dbl_t A_tile[16 * 16];
         dbl_t cRes[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

         const int aStart = widthA * 16 * blockIdx.y;
         const int aEnd = aStart + widthA - 1;
         const int bStart = 16 * 4 * blockIdx.x;
         const int bStep = 16 * widthB;

         int cIdx = widthB * 16 * blockIdx.y + 16 * blockIdx.x * 4
                         + 16 * threadIdx.y + threadIdx.x;

         for (int aIdx = aStart, bIdx = bStart; aIdx <= aEnd;
              aIdx += 16, bIdx += bStep)
         {

           #pragma unroll
           for (int i = 0; i < 16; i += 4)
           {
             const int aIndex = aIdx + widthA * (i + threadIdx.y) + threadIdx.x;
             if (aIndex < maxA)
               A_tile[i + threadIdx.y + 16 * threadIdx.x] = A[aIndex];
           }

           __syncthreads();

           if ((bStart + threadIdx.x + 16 * threadIdx.y) < 75)
           {
              const int bPIndex = bIdx + 16 * threadIdx.y + threadIdx.x;
              const dbl_t *bPartial = &B[bPIndex];
              int indexPartial = 0;
              int tileIndex = 0;
              const int maxPartial = maxB - bPIndex;

              #pragma unroll
              for (int i = 0; i < 16; ++i)
              {
                if (indexPartial < maxPartial)
                {
                  const dbl_t bVal = bPartial[indexPartial];

                  #pragma unroll
                  for (int j = 0; j < 16; ++j)
                    cRes[j] += A_tile[tileIndex + j] * bVal;
                  tileIndex += 16;
                  indexPartial += widthB;
                }
              }
           }
           __syncthreads();
         }

         if ((bStart + threadIdx.x + 16 * threadIdx.y) >= 75)
           return;

         #pragma unroll
         for (int i = 0; i < 16; ++i)
         {
           if (cIdx < maxC)
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

      template<int hSize, int wSize, int chSize, int nbFilter>
      __global__
      void rot_ker_transform_cuda(const dbl_t *ker, dbl_t *res)
      {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int filterIdx = index % nbFilter;
        const int chIdx = ((index - filterIdx) % (chSize * nbFilter))/nbFilter;
        const int wIdx = ((index - filterIdx - (chIdx * nbFilter))
                      % (chSize * nbFilter * wSize))/(chSize * nbFilter);
        const int hIdx = (index - wIdx * chSize * nbFilter - chIdx * nbFilter
                        - filterIdx)/(wSize * chSize * nbFilter);
        /*const int nIndex = filterIdx * hSize * wSize * chSize
                + chIdx * wSize * hSize + hIdx * wSize + wIdx;*/
        const int transpose_hIdx = hSize - hIdx - 1;
        const int transpose_wIdx = wSize - wIdx - 1;
        const int nIndex = chIdx * hSize * wSize * nbFilter
                           + filterIdx * wSize * hSize + transpose_hIdx * wSize
                           + transpose_wIdx;
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

    template<int P, int Q, int depthSize, int width, int height,
             int widthK, int heightK, int stride, int padL, int padTop,
             int batchSize, int chSize>
    __global__ void
    im2col_trans(float* res,
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
        data += z * (batchSize * width * height)
                + (yOldIdx * batchSize * width)
                + xOldIdx * batchSize + batchIdx;

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
              *res = data[dj * batchSize * width + di * batchSize];
            else
              *res = 0;
            res += (P*Q*batchSize);
          }
        }
      }
    }

    template<int sOutTot0, int sOutTot1, int stride, int padL, int padTop,
             int hSize, int wSize, int chSize>
    __global__
    void padd_full_conv(const dbl_t *data, dbl_t *res, const int batchIdx)
    {
      int hIdx = blockIdx.y * 4 + threadIdx.y;
      int wIdx = blockIdx.x * 16 + threadIdx.x;
      const int nhIdx = hIdx * stride + padTop;
      const int chIdx = wIdx % chSize;
      const int rwIdx = wIdx / chSize;

      res[batchIdx * sOutTot0 + nhIdx * sOutTot1
          + (rwIdx * stride + padL) * chSize + chIdx] =
          data[batchIdx * hSize * wSize * chSize
               + hIdx * wSize * chSize
               + wIdx];
    }

    template<int sOutTot0, int sOutTot1, int hSize,
            int chSize, int wSize, int stride>
    __global__
    void padd_ker_rot(const dbl_t *data, dbl_t *res, const int batchIdx)
    {
      int hIdx = blockIdx.y * 4 + threadIdx.y;
      int wIdx = blockIdx.x * 16 + threadIdx.x;
      const int nhIdx = hIdx * stride;
      const int chIdx = wIdx % chSize;
      const int rwIdx = wIdx / chSize;

      /*res[batchIdx * sOutTot0 + nhIdx * sOutTot1
          + (rwIdx * stride) * chSize + chIdx] =
          data[rwIdx * batchSize * hSize * chSize
               + batchIdx * hSize * chSize
               + hIdx * chSize + chIdx];*/
      res[nhIdx * sOutTot0 + (rwIdx * stride) * sOutTot1
          + batchIdx * chSize + chIdx] =
          data[batchIdx * hSize * wSize * chSize
               + hIdx * wSize * chSize + rwIdx * chSize + chIdx];
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

    template<int width, int P, int Q, int nbFilter, int blSize>
    __global__
    void transform_res_back(const dbl_t *resConv, dbl_t *transf)
    {
        const int blkIdx = blockIdx.y * width * blSize + blockIdx.x * 16;
        const int thIdx = threadIdx.y * width + threadIdx.x;
        const int realIdx = blkIdx + thIdx;
        const int nbFIdx = blockIdx.y * blSize + threadIdx.y;//realIdx / width;
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

    template<int width, int P, int Q, int nbFilter, int nbChan, int blSize>
    __global__
    void transform_res_ker(const dbl_t *resConv, dbl_t *transf)
    {
        const int blkIdx = blockIdx.y * width * 4 + blockIdx.x * blSize;
        const int thIdx = threadIdx.y * width + threadIdx.x;
        const int realIdx = blkIdx + thIdx;
        const int nbFIdx = blockIdx.y * 4 + threadIdx.y;//realIdx / width;
        const int tmp0 = nbFIdx * width;
        const int batchIdx = (realIdx - tmp0) / (P * Q);
        const int tmp1 = batchIdx * P * Q;
        const int tmp2 = realIdx - tmp0 - tmp1;
        const int hIdx = tmp2 / P;
        const int wIdx = tmp2 % P;

        transf[hIdx * Q * nbChan * nbFilter + wIdx * nbChan * nbFilter
                + batchIdx * nbFilter + nbFIdx] = resConv[realIdx];
    }

    template<int width, int P, int Q, int nbFilter, int nbChan, int blSize>
    __global__
    void transform_res_ker15(const dbl_t *resConv, dbl_t *transf)
    {
        if (blockIdx.x == 4 && threadIdx.x >= 11)
          return;
        const int blkIdx = blockIdx.y * width * 4 + blockIdx.x * blSize;
        const int thIdx = threadIdx.y * width + threadIdx.x;
        const int realIdx = blkIdx + thIdx;
        const int nbFIdx = blockIdx.y * 4 + threadIdx.y;//realIdx / width;
        const int tmp0 = nbFIdx * width;
        const int batchIdx = (realIdx - tmp0) / (P * Q);
        const int tmp1 = batchIdx * P * Q;
        const int tmp2 = realIdx - tmp0 - tmp1;
        const int hIdx = tmp2 / P;
        const int wIdx = tmp2 % P;

        transf[hIdx * Q * nbChan * nbFilter + wIdx * nbChan * nbFilter
                + batchIdx * nbFilter + nbFIdx] = resConv[realIdx];
    }

    void conv2d_d0_caller(const float *data, const float *ker, float *res)
    {
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);

      constexpr int totalKernelSize = 5 * 5 * 3 * 64;
      dbl_t *newKernelCuda = tensor_alloc(totalKernelSize);


      dim3 dimGrid(totalKernelSize / 32);
      dim3 dimBlock(32);
      ker_transform_cuda<5, 5, 3, 64><<<dimGrid, dimBlock>>>(ker, newKernelCuda);

      cudaDeviceSynchronize();

      constexpr int newInputSize = 3 * 5 * 5 * 64 * 32 * 32;
      dbl_t *newInputCuda = tensor_alloc(newInputSize);

      #pragma unroll
      for (int b = 0; b < 64; ++b)
      {
        im2col<32, 32, 3072, 64, 64, 5, 5, 2, 1, 1, 64, 3>
                <<<3, 1024>>>(newInputCuda, data, b);
      }

      cudaDeviceSynchronize();


      constexpr int resSize = 64 * 64 * 32 * 32;
      dbl_t *resConvCuda = tensor_alloc(resSize);
      dim3 dimBlockConv(16, 4);
      dim3 dimGridConv(1024, 4);
      mat_mul_cuda16<75, 65536, 4800, 4915200><<<dimGridConv, dimBlockConv>>>(
                      newKernelCuda, newInputCuda, resConvCuda);

      cudaDeviceSynchronize();

      dim3 dimGridTransf(4096, 16);
      dim3 dimBlockTransf(16, 4);
      transform_res<65536, 32, 32, 64><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

      //cudaDeviceSynchronize();
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float time;
      cudaEventElapsedTime(&time, start, stop);
      std::ofstream fos("time_gemm.log", std::ios::app);
      fos << "time (fwd_gemm_d0) = " << time << "ms\n" << std::endl;

      tensor_free(newKernelCuda);
      tensor_free(newInputCuda);
      tensor_free(resConvCuda);
    }

    void conv2d_d1_caller(const float *data, const float *ker, float *res)
    {
      cudaEvent_t start, stop, stopKer, startImg, stopImg,
                  startMul, stopMul, startTransf;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventCreate(&stopKer);
      cudaEventCreate(&startImg);
      cudaEventCreate(&stopImg);
      cudaEventCreate(&startMul);
      cudaEventCreate(&stopMul);
      cudaEventCreate(&startTransf);
      cudaEventRecord(start, 0);

      
      constexpr int totalKernelSize1 = 5 * 5 * 128 * 64;
      dbl_t *newKernelCuda = tensor_alloc(totalKernelSize1);
      dim3 dimGrid(totalKernelSize1 / 32);
      dim3 dimBlock(32);
      ker_transform_cuda<5, 5, 64, 128><<<dimGrid, dimBlock>>>(ker, newKernelCuda);

      //cudaDeviceSynchronize();
      cudaEventRecord(stopKer, 0);
      cudaEventSynchronize(stopKer);
      float timeKer;
      cudaEventElapsedTime(&timeKer, start, stopKer);

      cudaEventRecord(startImg, 0);
      constexpr int newInputSize1 = 64 * 5 * 5 * 64 * 16 * 16;
      dbl_t *newInputCuda = tensor_alloc(newInputSize1);

      #pragma unroll
      for (int b = 0; b < 64; ++b)
      {
        im2col<16, 16, 16384, 32, 32, 5, 5, 2, 1, 1, 64, 64>
                <<<16, 1024>>>(newInputCuda, data, b);
      }

      //cudaDeviceSynchronize();
      cudaEventRecord(stopImg, 0);
      cudaEventSynchronize(stopImg);
      float timeImg;
      cudaEventElapsedTime(&timeImg, startImg, stopImg);

      cudaEventRecord(startMul, 0);
      constexpr int resSize1 = 128 * 64 * 16 * 16;
      dbl_t *resConvCuda = tensor_alloc(resSize1);
      dim3 dimBlockConv(32, 4);
      dim3 dimGridConv(128, 4);
      mat_mul_cuda32<1600, 16384, 204800, 26214400><<<dimGridConv, dimBlockConv>>>(
                      newKernelCuda, newInputCuda, resConvCuda);

      //cudaDeviceSynchronize();
      cudaEventRecord(stopMul, 0);
      cudaEventSynchronize(stopMul);
      float timeMul;
      cudaEventElapsedTime(&timeMul, startMul, stopMul);

      cudaEventRecord(startTransf, 0);
      dim3 dimGridTransf(1024, 32);
      dim3 dimBlockTransf(16, 4);
      transform_res<16384, 16, 16, 128><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

      //cudaDeviceSynchronize();
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float timeTransf;
      cudaEventElapsedTime(&timeTransf, startTransf, stop);

      float time;
      cudaEventElapsedTime(&time, start, stop);
      std::ofstream fos("time_gemm.log", std::ios::app);
      fos << "time (fwd_gemm_d1_ker) = " << timeKer << "ms\n" << std::endl;
      fos << "time (fwd_gemm_d1_img) = " << timeImg << "ms\n" << std::endl;
      fos << "time (fwd_gemm_d1_mul) = " << timeMul << "ms\n" << std::endl;
      fos << "time (fwd_gemm_d1_transf) = " << timeTransf << "ms\n" << std::endl;
      fos << "time (fwd_gemm_d1_all) = " << time << "ms\n" << std::endl;

      tensor_free(newKernelCuda);
      tensor_free(newInputCuda);
      tensor_free(resConvCuda);
    }

    void conv2d_d2_caller(const float *data, const float *ker, float *res)
    {
      cudaEvent_t start, stop, stopKer, startImg, stopImg,
                  startMul, stopMul, startTransf;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventCreate(&stopKer);
      cudaEventCreate(&startImg);
      cudaEventCreate(&stopImg);
      cudaEventCreate(&startMul);
      cudaEventCreate(&stopMul);
      cudaEventCreate(&startTransf);
      cudaEventRecord(start, 0);

      constexpr int totalKernelSize1 = 5 * 5 * 128 * 256;
      dbl_t *newKernelCuda = tensor_alloc(totalKernelSize1);
      dim3 dimGrid(totalKernelSize1 / 32);
      dim3 dimBlock(32);
      ker_transform_cuda<5, 5, 128, 256><<<dimGrid, dimBlock>>>(ker, newKernelCuda);

      //cudaDeviceSynchronize();
      cudaEventRecord(stopKer, 0);
      cudaEventSynchronize(stopKer);
      float timeKer;
      cudaEventElapsedTime(&timeKer, start, stopKer);

      cudaEventRecord(startImg, 0);
      constexpr int newInputSize1 = 5 * 5 * 128 * 64 * 8 * 8;
      dbl_t *newInputCuda = tensor_alloc(newInputSize1);

      #pragma unroll
      for (int b = 0; b < 64; ++b)
      {
        im2col<8, 8, 8192, 16, 16, 5, 5, 2, 1, 1, 64, 128>
                <<<8, 1024>>>(newInputCuda, data, b);
      }

      //cudaDeviceSynchronize();
      cudaEventRecord(stopImg, 0);
      cudaEventSynchronize(stopImg);
      float timeImg;
      cudaEventElapsedTime(&timeImg, startImg, stopImg);

      cudaEventRecord(startMul, 0);
      constexpr int resSize1 = 256 * 64 * 8 * 8;
      dbl_t *resConvCuda = tensor_alloc(resSize1);
      dim3 dimBlockConv(16, 4);
      dim3 dimGridConv(64, 16);
      mat_mul_cuda16<3200, 4096, 819200, 13107200><<<dimGridConv, dimBlockConv>>>(
                      newKernelCuda, newInputCuda, resConvCuda);

      //cudaDeviceSynchronize();
      cudaEventRecord(stopMul, 0);
      cudaEventSynchronize(stopMul);
      float timeMul;
      cudaEventElapsedTime(&timeMul, startMul, stopMul);

      cudaEventRecord(startTransf, 0);
      dim3 dimGridTransf(256, 64);
      dim3 dimBlockTransf(16, 4);
      transform_res<4096, 8, 8, 256><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

      //cudaDeviceSynchronize();
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float timeTransf;
      cudaEventElapsedTime(&timeTransf, startTransf, stop);

      float time;
      cudaEventElapsedTime(&time, start, stop);
      std::ofstream fos("time_gemm.log", std::ios::app);
      fos << "time (fwd_gemm_d2_ker) = " << timeKer << "ms\n" << std::endl;
      fos << "time (fwd_gemm_d2_img) = " << timeImg << "ms\n" << std::endl;
      fos << "time (fwd_gemm_d2_mul) = " << timeMul << "ms\n" << std::endl;
      fos << "time (fwd_gemm_d2_transf) = " << timeTransf << "ms\n" << std::endl;
      fos << "time (fwd_gemm_d2_all) = " << time << "ms\n" << std::endl;

      tensor_free(newKernelCuda);
      tensor_free(newInputCuda);
      tensor_free(resConvCuda);
    }

    void conv2d_d3_caller(const float *data, const float *ker, float *res)
    {
      cudaEvent_t start, stop, stopKer, startImg, stopImg,
                  startMul, stopMul, startTransf;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventCreate(&stopKer);
      cudaEventCreate(&startImg);
      cudaEventCreate(&stopImg);
      cudaEventCreate(&startMul);
      cudaEventCreate(&stopMul);
      cudaEventCreate(&startTransf);
      cudaEventRecord(start, 0);

      constexpr int totalKernelSize1 = 5 * 5 * 256 * 512;
      dbl_t *newKernelCuda = tensor_alloc(totalKernelSize1);
      dim3 dimGrid(totalKernelSize1 / 32);
      dim3 dimBlock(32);
      ker_transform_cuda<5, 5, 256, 512><<<dimGrid, dimBlock>>>(ker, newKernelCuda);

      //cudaDeviceSynchronize();
      cudaEventRecord(stopKer, 0);
      cudaEventSynchronize(stopKer);
      float timeKer;
      cudaEventElapsedTime(&timeKer, start, stopKer);

      cudaEventRecord(startImg, 0);
      constexpr int newInputSize1 = 5 * 5 * 256 * 64 * 4 * 4;
      dbl_t *newInputCuda = tensor_alloc(newInputSize1);

      #pragma unroll
      for (int b = 0; b < 64; ++b)
      {
        im2col<4, 4, 4096, 8, 8, 5, 5, 2, 1, 1, 64, 256>
                <<<4, 1024>>>(newInputCuda, data, b);
      }

      //cudaDeviceSynchronize();
      cudaEventRecord(stopImg, 0);
      cudaEventSynchronize(stopImg);
      float timeImg;
      cudaEventElapsedTime(&timeImg, startImg, stopImg);

      cudaEventRecord(startMul, 0);
      constexpr int resSize1 = 512 * 64 * 4 * 4;
      dbl_t *resConvCuda = tensor_alloc(resSize1);
      dim3 dimBlockConv(16, 4);
      dim3 dimGridConv(16, 32);
      mat_mul_cuda16<6400, 1024, 3276800, 6553600><<<dimGridConv, dimBlockConv>>>(
                      newKernelCuda, newInputCuda, resConvCuda);

      //cudaDeviceSynchronize();
      cudaEventRecord(stopMul, 0);
      cudaEventSynchronize(stopMul);
      float timeMul;
      cudaEventElapsedTime(&timeMul, startMul, stopMul);

      cudaEventRecord(startTransf, 0);
      dim3 dimGridTransf(64, 128);
      dim3 dimBlockTransf(16, 4);
      transform_res<1024, 4, 4, 512><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

      //cudaDeviceSynchronize();
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float timeTransf;
      cudaEventElapsedTime(&timeTransf, startTransf, stop);

      float time;
      cudaEventElapsedTime(&time, start, stop);
      std::ofstream fos("time_gemm.log", std::ios::app);
      fos << "time (fwd_gemm_d3_ker) = " << timeKer << "ms\n" << std::endl;
      fos << "time (fwd_gemm_d3_img) = " << timeImg << "ms\n" << std::endl;
      fos << "time (fwd_gemm_d3_mul) = " << timeMul << "ms\n" << std::endl;
      fos << "time (fwd_gemm_d3_transf) = " << timeTransf << "ms\n" << std::endl;
      fos << "time (fwd_gemm_d3_all) = " << time << "ms\n" << std::endl;

      tensor_free(newKernelCuda);
      tensor_free(newInputCuda);
      tensor_free(resConvCuda);
    }
  }

  void conv2d_d0_dx_caller(const float *data, const float *ker, float *res)
  {
    cudaEvent_t start, stop, stopImg, stopKer;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stopImg);
    cudaEventCreate(&stopKer);
    cudaEventRecord(start, 0);

    constexpr int paddFullSize = 64 * 68 * 68 * 64;
    dbl_t *paddFullCuda = tensor_alloc(paddFullSize);
    dim3 dimGridPadd(128, 8);
    dim3 dimBlockPadd(16, 4);

    #pragma unroll
    for(int b = 0; b < 64; ++b)
    {
      padd_full_conv<68*68*64, 68*64, 2, 3, 3, 32, 32, 64>
              <<<dimGridPadd, dimBlockPadd>>>(data, paddFullCuda, b);
    }


    constexpr int newInputSize = 64 * 5 * 5 * 64 * 64 * 64;
    dbl_t *newInputCuda = tensor_alloc(newInputSize);

    #pragma unroll
    for (int b = 0; b < 64; ++b)
    {
      im2col<64, 64, 262144, 68, 68, 5, 5, 1, 0, 0, 64, 64>
              <<<256, 1024>>>(newInputCuda, paddFullCuda, b);
    }

    cudaEventRecord(stopImg, 0);
    cudaEventSynchronize(stopImg);
    float timeImg;
    cudaEventElapsedTime(&timeImg, start, stopImg);

    //cudaDeviceSynchronize();

    constexpr int totalKernelSize = 5 * 5 * 3 * 64;
    dbl_t *newKernelCuda = tensor_alloc(totalKernelSize);
    dim3 dimGrid(totalKernelSize / 32);
    dim3 dimBlock(32);
    rot_ker_transform_cuda<5, 5, 3, 64><<<dimGrid, dimBlock>>>(ker, newKernelCuda);

    //cudaDeviceSynchronize();
    cudaEventRecord(stopKer, 0);
    cudaEventSynchronize(stopKer);
    float timeKer;
    cudaEventElapsedTime(&timeKer, start, stopKer);

    //gpuErrchk(cudaPeekAtLastError());

    constexpr int resSize1 = 64 * 64 * 64 * 3;
    dbl_t *resConvCuda = tensor_alloc(resSize1);
    dim3 dimBlockConv(16, 4);
    dim3 dimGridConv(4096, 1);
    back_mat_mul_cuda16<1600, 262144, 4800, 419430400, 786432><<<dimGridConv, dimBlockConv>>>(
                    newKernelCuda, newInputCuda, resConvCuda);

    //gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaDeviceSynchronize());
    cudaDeviceSynchronize();

    dim3 dimGridTransf(16384, 1);
    dim3 dimBlockTransf(16, 3);
    transform_res_back<262144, 64, 64, 3, 3><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    //gpuErrchk(cudaPeekAtLastError());

    std::ofstream fos("time_gemm_dx.log", std::ios::app);
    fos << "time (dx_gemm_d0_Img) = " << timeImg << "ms\n";
    fos << "time (dx_gemm_d0_Ker) = " << timeKer << "ms\n";
    fos << "time (dx_gemm_d0_all) = " << time << "ms\n" << std::endl;

    tensor_free(newKernelCuda);
    tensor_free(newInputCuda);
    tensor_free(resConvCuda);
    tensor_free(paddFullCuda);
  }

  void conv2d_d1_dx_caller(const float *data, const float *ker, float *res)
  {
    cudaEvent_t start, stop, stopImg, stopKer;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stopImg);
    cudaEventCreate(&stopKer);
    cudaEventRecord(start, 0);

    constexpr int paddFullSize = 64 * 36 * 36 * 128;
    dbl_t *paddFullCuda = tensor_alloc(paddFullSize);
    dim3 dimGridPadd(128, 4);
    dim3 dimBlockPadd(16, 4);

    #pragma unroll
    for(int b = 0; b < 64; ++b)
    {
      padd_full_conv<36*36*128, 36*128, 2, 3, 3, 16, 16, 128>
              <<<dimGridPadd, dimBlockPadd>>>(data, paddFullCuda, b);
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    //cudaDeviceSynchronize();

    constexpr int newInputSize = 128 * 5 * 5 * 64 * 32 * 32;
    dbl_t *newInputCuda = tensor_alloc(newInputSize);

    #pragma unroll
    for (int b = 0; b < 64; ++b)
    {
      im2col<32, 32, 131072, 36, 36, 5, 5, 1, 0, 0, 64, 128>
              <<<128, 1024>>>(newInputCuda, paddFullCuda, b);
    }

    cudaEventRecord(stopImg, 0);
    cudaEventSynchronize(stopImg);
    float timeImg;
    cudaEventElapsedTime(&timeImg, start, stopImg);
    gpuErrchk(cudaPeekAtLastError());

    //cudaDeviceSynchronize();
    constexpr int totalKernelSize = 5 * 5 * 64 * 128;
    dbl_t *newKernelCuda = tensor_alloc(totalKernelSize);
    dim3 dimGrid(totalKernelSize / 32);
    dim3 dimBlock(32);
    rot_ker_transform_cuda<5, 5, 64, 128><<<dimGrid, dimBlock>>>(ker, newKernelCuda);

    //cudaDeviceSynchronize();
    cudaEventRecord(stopKer, 0);
    cudaEventSynchronize(stopKer);
    float timeKer;
    cudaEventElapsedTime(&timeKer, start, stopKer);

    //gpuErrchk(cudaPeekAtLastError());

    constexpr int resSize1 = 64 * 32 * 32 * 64;
    dbl_t *resConvCuda = tensor_alloc(resSize1);
    dim3 dimBlockConv(16, 4);
    dim3 dimGridConv(1024, 4);
    back_mat_mul_cuda16<3200, 65536, 204800, 209715200, 4194304><<<dimGridConv, dimBlockConv>>>(
                    newKernelCuda, newInputCuda, resConvCuda);

    //gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaDeviceSynchronize());
    cudaDeviceSynchronize();

    dim3 dimGridTransf(4096, 16);
    dim3 dimBlockTransf(16, 4);
    transform_res_back<65536, 32, 32, 64, 4><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    //gpuErrchk(cudaPeekAtLastError());

    std::ofstream fos("time_gemm_dx.log", std::ios::app);
    fos << "time (dx_gemm_d1_Img) = " << timeImg << "ms\n";
    fos << "time (dx_gemm_d1_Ker) = " << timeKer << "ms\n";
    fos << "time (dx_gemm_d1_all) = " << time << "ms\n" << std::endl;

    tensor_free(newKernelCuda);
    tensor_free(newInputCuda);
    tensor_free(resConvCuda);
    tensor_free(paddFullCuda);
  }

  void conv2d_d2_dx_caller(const float *data, const float *ker, float *res)
  {
    cudaEvent_t start, stop, stopImg, stopKer;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stopImg);
    cudaEventCreate(&stopKer);
    cudaEventRecord(start, 0);

    constexpr int paddFullSize = 64 * 20 * 20 * 256;
    dbl_t *paddFullCuda = tensor_alloc(paddFullSize);
    dim3 dimGridPadd(128, 2);
    dim3 dimBlockPadd(16, 4);

    #pragma unroll
    for(int b = 0; b < 64; ++b)
    {
      padd_full_conv<20*20*256, 20*256, 2, 3, 3, 8, 8, 256>
              <<<dimGridPadd, dimBlockPadd>>>(data, paddFullCuda, b);
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    //cudaDeviceSynchronize();

    constexpr int newInputSize = 256 * 5 * 5 * 64 * 16 * 16;
    dbl_t *newInputCuda = tensor_alloc(newInputSize);

    #pragma unroll
    for (int b = 0; b < 64; ++b)
    {
      im2col<16, 16, 65536, 20, 20, 5, 5, 1, 0, 0, 64, 256>
              <<<64, 1024>>>(newInputCuda, paddFullCuda, b);
    }

    cudaEventRecord(stopImg, 0);
    cudaEventSynchronize(stopImg);
    float timeImg;
    cudaEventElapsedTime(&timeImg, start, stopImg);
    gpuErrchk(cudaPeekAtLastError());

    //cudaDeviceSynchronize();

    constexpr int totalKernelSize = 5 * 5 * 128 * 256;
    dbl_t *newKernelCuda = tensor_alloc(totalKernelSize);
    dim3 dimGrid(totalKernelSize / 32);
    dim3 dimBlock(32);
    rot_ker_transform_cuda<5, 5, 128, 256><<<dimGrid, dimBlock>>>(ker, newKernelCuda);

    //cudaDeviceSynchronize();
    cudaEventRecord(stopKer, 0);
    cudaEventSynchronize(stopKer);
    float timeKer;
    cudaEventElapsedTime(&timeKer, start, stopKer);

    //gpuErrchk(cudaPeekAtLastError());

    constexpr int resSize1 = 64 * 16 * 16 * 128;
    dbl_t *resConvCuda = tensor_alloc(resSize1);
    dim3 dimBlockConv(16, 4);
    dim3 dimGridConv(256, 8);
    back_mat_mul_cuda16<6400, 16384, 819200, 104857600, 2097152><<<dimGridConv, dimBlockConv>>>(
                    newKernelCuda, newInputCuda, resConvCuda);

    //gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaDeviceSynchronize());
    cudaDeviceSynchronize();

    dim3 dimGridTransf(1024, 32);
    dim3 dimBlockTransf(16, 4);
    transform_res_back<16384, 16, 16, 128, 4><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    //gpuErrchk(cudaPeekAtLastError());

    std::ofstream fos("time_gemm_dx.log", std::ios::app);
    fos << "time (dx_gemm_d2_Img) = " << timeImg << "ms\n";
    fos << "time (dx_gemm_d2_Ker) = " << timeKer << "ms\n";
    fos << "time (dx_gemm_d2_all) = " << time << "ms\n" << std::endl;

    tensor_free(newKernelCuda);
    tensor_free(newInputCuda);
    tensor_free(resConvCuda);
    tensor_free(paddFullCuda);
  }

  void conv2d_d3_dx_caller(const float *data, const float *ker, float *res)
  {
    cudaEvent_t start, stop, stopImg, stopKer;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stopImg);
    cudaEventCreate(&stopKer);
    cudaEventRecord(start, 0);

    constexpr int paddFullSize = 64 * 12 * 12 * 512;
    dbl_t *paddFullCuda = tensor_alloc(paddFullSize);
    dim3 dimGridPadd(128, 1);
    dim3 dimBlockPadd(16, 4);

    #pragma unroll
    for(int b = 0; b < 64; ++b)
    {
      padd_full_conv<12*12*512, 12*512, 2, 3, 3, 4, 4, 512>
              <<<dimGridPadd, dimBlockPadd>>>(data, paddFullCuda, b);
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    //cudaDeviceSynchronize();

    constexpr int newInputSize = 512 * 5 * 5 * 64 * 8 * 8;
    dbl_t *newInputCuda = tensor_alloc(newInputSize);

    #pragma unroll
    for (int b = 0; b < 64; ++b)
    {
      im2col<8, 8, 32768, 12, 12, 5, 5, 1, 0, 0, 64, 512>
              <<<32, 1024>>>(newInputCuda, paddFullCuda, b);
    }

    cudaEventRecord(stopImg, 0);
    cudaEventSynchronize(stopImg);
    float timeImg;
    cudaEventElapsedTime(&timeImg, start, stopImg);
    gpuErrchk(cudaPeekAtLastError());

    //cudaDeviceSynchronize();

    constexpr int totalKernelSize = 5 * 5 * 256 * 512;
    dbl_t *newKernelCuda = tensor_alloc(totalKernelSize);
    dim3 dimGrid(totalKernelSize / 32);
    dim3 dimBlock(32);
    rot_ker_transform_cuda<5, 5, 256, 512><<<dimGrid, dimBlock>>>(ker, newKernelCuda);

    //cudaDeviceSynchronize();
    cudaEventRecord(stopKer, 0);
    cudaEventSynchronize(stopKer);
    float timeKer;
    cudaEventElapsedTime(&timeKer, start, stopKer);

    //gpuErrchk(cudaPeekAtLastError());

    constexpr int resSize1 = 64 * 8 * 8 * 256;
    dbl_t *resConvCuda = tensor_alloc(resSize1);
    dim3 dimBlockConv(16, 4);
    dim3 dimGridConv(64, 16);
    back_mat_mul_cuda16<12800, 4096, 3276800, 52428800, 1048576><<<dimGridConv, dimBlockConv>>>(
                    newKernelCuda, newInputCuda, resConvCuda);

    //gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaDeviceSynchronize());
    cudaDeviceSynchronize();

    dim3 dimGridTransf(256, 64);
    dim3 dimBlockTransf(16, 4);
    transform_res_back<4096, 8, 8, 256, 4><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    //gpuErrchk(cudaPeekAtLastError());

    std::ofstream fos("time_gemm_dx.log", std::ios::app);
    fos << "time (dx_gemm_d3_Img) = " << timeImg << "ms\n";
    fos << "time (dx_gemm_d3_Ker) = " << timeKer << "ms\n";
    fos << "time (dx_gemm_d3_all) = " << time << "ms\n" << std::endl;

    tensor_free(newKernelCuda);
    tensor_free(newInputCuda);
    tensor_free(resConvCuda);
    tensor_free(paddFullCuda);
  }

  void conv2d_d0_dk_caller(const float *data, const float *ker, float *res)
  {
    cudaEvent_t start, stop, stopImg, stopKer;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stopImg);
    cudaEventCreate(&stopKer);
    cudaEventRecord(start, 0);

    constexpr int paddFullSize = 63 * 63 * 64 * 64;
    dbl_t *paddFullCuda = tensor_alloc(paddFullSize);
    dbl_t *zeroVal = (dbl_t*)calloc(paddFullSize, sizeof(dbl_t));
    cudaMemcpy(paddFullCuda, zeroVal, sizeof(dbl_t) * paddFullSize, cudaMemcpyHostToDevice);
    free(zeroVal);
    dim3 dimGridPadd(128, 8);
    dim3 dimBlockPadd(16, 4);

    #pragma unroll
    for(int b = 0; b < 64; ++b)
    {
      padd_ker_rot<63*64*64, 64*64, 32, 64, 32, 2>
              <<<dimGridPadd, dimBlockPadd>>>(ker, paddFullCuda, b);
    }

    //cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    constexpr int newInputSize = 64 * 63 * 63 * 3 * 5 * 5;
    dbl_t *newInputCuda = tensor_alloc(newInputSize);

    #pragma unroll
    for (int b = 0; b < 3; ++b)
    {
      im2col_trans<5, 5, 1600, 64, 64, 63, 63, 1, 1, 1, 3, 64>
              <<<25, 64>>>(newInputCuda, data, b);
    }

    cudaEventRecord(stopImg, 0);
    cudaEventSynchronize(stopImg);
    float timeImg;
    cudaEventElapsedTime(&timeImg, start, stopImg);

    //cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    constexpr int totalKernelSize = 63 * 63 * 64 * 64;
    dbl_t *newKernelCuda = tensor_alloc(totalKernelSize);
    dim3 dimGrid(totalKernelSize / 32);
    dim3 dimBlock(32);
    ker_transform_cuda<63, 63, 64, 64><<<dimGrid, dimBlock>>>(paddFullCuda, newKernelCuda);

    //cudaDeviceSynchronize();
    cudaEventRecord(stopKer, 0);
    cudaEventSynchronize(stopKer);
    float timeKer;
    cudaEventElapsedTime(&timeKer, start, stopKer);

    gpuErrchk(cudaPeekAtLastError());


    constexpr int resSize1 = 64 * 5 * 5 * 3;
    dbl_t *resConvCuda = tensor_alloc(resSize1);
    dim3 dimBlockConv(16, 4);
    dim3 dimGridConv(2, 4);
    back_mat_mul_cuda75<254016, 75, 16257024, 19051200, 4800><<<dimGridConv, dimBlockConv>>>(
                    newKernelCuda, newInputCuda, resConvCuda);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    //cudaDeviceSynchronize();

    dim3 dimGridTransf(5, 16);
    dim3 dimBlockTransf(15, 4);
    transform_res_ker<75, 5, 5, 64, 3, 15><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    gpuErrchk(cudaPeekAtLastError());

    std::ofstream fos("time_gemm_dk.log", std::ios::app);
    fos << "time (dk_gemm_d0_Img) = " << timeImg << "ms\n";
    fos << "time (dk_gemm_d0_Ker) = " << timeKer << "ms\n";
    fos << "time (dk_gemm_d0_all) = " << time << "ms\n" << std::endl;

    tensor_free(newKernelCuda);
    tensor_free(newInputCuda);
    tensor_free(resConvCuda);
    tensor_free(paddFullCuda);
  }

  void conv2d_d1_dk_caller(const float *data, const float *ker, float *res)
  {
    cudaEvent_t start, stop, stopImg, stopKer;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stopImg);
    cudaEventCreate(&stopKer);
    cudaEventRecord(start, 0);

    constexpr int paddFullSize = 31 * 31 * 128 * 64;
    dbl_t *paddFullCuda = tensor_alloc(paddFullSize);
    dbl_t *zeroVal = (dbl_t*)calloc(paddFullSize, sizeof(dbl_t));
    cudaMemcpy(paddFullCuda, zeroVal, sizeof(dbl_t) * paddFullSize, cudaMemcpyHostToDevice);
    free(zeroVal);
    dim3 dimGridPadd(128, 4);
    dim3 dimBlockPadd(16, 4);

    #pragma unroll
    for(int b = 0; b < 64; ++b)
    {
      padd_ker_rot<31*128*64, 128*64, 16, 128, 16, 2>
              <<<dimGridPadd, dimBlockPadd>>>(ker, paddFullCuda, b);
    }

    //cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    constexpr int newInputSize = 64 * 31 * 31 * 64 * 5 * 5;
    dbl_t *newInputCuda = tensor_alloc(newInputSize);

    #pragma unroll
    for (int b = 0; b < 64; ++b)
    {
      im2col_trans<5, 5, 1600, 32, 32, 31, 31, 1, 1, 1, 64, 64>
              <<<25, 64>>>(newInputCuda, data, b);
    }

    cudaEventRecord(stopImg, 0);
    cudaEventSynchronize(stopImg);
    float timeImg;
    cudaEventElapsedTime(&timeImg, start, stopImg);

    //cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    constexpr int totalKernelSize = 31 * 31 * 64 * 128;
    dbl_t *newKernelCuda = tensor_alloc(totalKernelSize);
    dim3 dimGrid(totalKernelSize / 32);
    dim3 dimBlock(32);
    ker_transform_cuda<31, 31, 64, 128><<<dimGrid, dimBlock>>>(paddFullCuda, newKernelCuda);

    //cudaDeviceSynchronize();
    cudaEventRecord(stopKer, 0);
    cudaEventSynchronize(stopKer);
    float timeKer;
    cudaEventElapsedTime(&timeKer, start, stopKer);

    gpuErrchk(cudaPeekAtLastError());

    constexpr int resSize1 = 64 * 5 * 5 * 128;
    dbl_t *resConvCuda = tensor_alloc(resSize1);
    dim3 dimBlockConv(16, 4);
    dim3 dimGridConv(25, 8);
    back_mat_mul_cuda16<61504, 1600, 7872512, 98406400, 204800><<<dimGridConv, dimBlockConv>>>(
                    newKernelCuda, newInputCuda, resConvCuda);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    //cudaDeviceSynchronize();

    dim3 dimGridTransf(100, 32);
    dim3 dimBlockTransf(16, 4);
    transform_res_ker<1600, 5, 5, 128, 64, 16><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    gpuErrchk(cudaPeekAtLastError());

    std::ofstream fos("time_gemm_dk.log", std::ios::app);
    fos << "time (dk_gemm_d1_Img) = " << timeImg << "ms\n";
    fos << "time (dk_gemm_d1_Ker) = " << timeKer << "ms\n";
    fos << "time (dk_gemm_d1_all) = " << time << "ms\n" << std::endl;

    tensor_free(newKernelCuda);
    tensor_free(newInputCuda);
    tensor_free(resConvCuda);
    tensor_free(paddFullCuda);
  }

  void conv2d_d2_dk_caller(const float *data, const float *ker, float *res)
  {
    cudaEvent_t start, stop, stopImg, stopKer;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stopImg);
    cudaEventCreate(&stopKer);
    cudaEventRecord(start, 0);

    constexpr int paddFullSize = 15 * 15 * 256 * 64;
    dbl_t *paddFullCuda = tensor_alloc(paddFullSize);
    dbl_t *zeroVal = (dbl_t*)calloc(paddFullSize, sizeof(dbl_t));
    cudaMemcpy(paddFullCuda, zeroVal, sizeof(dbl_t) * paddFullSize, cudaMemcpyHostToDevice);
    free(zeroVal);
    dim3 dimGridPadd(128, 2);
    dim3 dimBlockPadd(16, 4);

    #pragma unroll
    for(int b = 0; b < 64; ++b)
    {
      padd_ker_rot<15*256*64, 256*64, 8, 256, 8, 2>
              <<<dimGridPadd, dimBlockPadd>>>(ker, paddFullCuda, b);
    }

    //cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    constexpr int newInputSize = 128 * 15 * 15 * 64 * 5 * 5;
    dbl_t *newInputCuda = tensor_alloc(newInputSize);

    #pragma unroll
    for (int b = 0; b < 128; ++b)
    {
      im2col_trans<5, 5, 1600, 16, 16, 15, 15, 1, 1, 1, 128, 64>
              <<<25, 64>>>(newInputCuda, data, b);
    }

    cudaEventRecord(stopImg, 0);
    cudaEventSynchronize(stopImg);
    float timeImg;
    cudaEventElapsedTime(&timeImg, start, stopImg);

    //cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    constexpr int totalKernelSize = 15 * 15 * 64 * 256;
    dbl_t *newKernelCuda = tensor_alloc(totalKernelSize);
    dim3 dimGrid(totalKernelSize / 32);
    dim3 dimBlock(32);
    ker_transform_cuda<15, 15, 64, 256><<<dimGrid, dimBlock>>>(paddFullCuda, newKernelCuda);

    //cudaDeviceSynchronize();
    cudaEventRecord(stopKer, 0);
    cudaEventSynchronize(stopKer);
    float timeKer;
    cudaEventElapsedTime(&timeKer, start, stopKer);

    gpuErrchk(cudaPeekAtLastError());

    constexpr int resSize1 = 256 * 5 * 5 * 128;
    dbl_t *resConvCuda = tensor_alloc(resSize1);
    dim3 dimBlockConv(16, 4);
    dim3 dimGridConv(50, 16);
    back_mat_mul_cuda16<14400, 3200, 3686400, 46080000, 819200><<<dimGridConv, dimBlockConv>>>(
                    newKernelCuda, newInputCuda, resConvCuda);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    //cudaDeviceSynchronize();

    dim3 dimGridTransf(200, 64);
    dim3 dimBlockTransf(16, 4);
    transform_res_ker<3200, 5, 5, 256, 128, 16><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    gpuErrchk(cudaPeekAtLastError());

    std::ofstream fos("time_gemm_dk.log", std::ios::app);
    fos << "time (dk_gemm_d2_Img) = " << timeImg << "ms\n";
    fos << "time (dk_gemm_d2_Ker) = " << timeKer << "ms\n";
    fos << "time (dk_gemm_d2_all) = " << time << "ms\n" << std::endl;

    tensor_free(newKernelCuda);
    tensor_free(newInputCuda);
    tensor_free(resConvCuda);
    tensor_free(paddFullCuda);
  }

  void conv2d_d3_dk_caller(const float *data, const float *ker, float *res)
  {
    cudaEvent_t start, stop, stopImg, stopKer;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stopImg);
    cudaEventCreate(&stopKer);
    cudaEventRecord(start, 0);

    constexpr int paddFullSize = 7 * 7 * 512 * 64;
    dbl_t *paddFullCuda = tensor_alloc(paddFullSize);
    dbl_t *zeroVal = (dbl_t*)calloc(paddFullSize, sizeof(dbl_t));
    cudaMemcpy(paddFullCuda, zeroVal, sizeof(dbl_t) * paddFullSize, cudaMemcpyHostToDevice);
    free(zeroVal);
    dim3 dimGridPadd(128, 1);
    dim3 dimBlockPadd(16, 4);

    #pragma unroll
    for(int b = 0; b < 64; ++b)
    {
      padd_ker_rot<7*512*64, 512*64, 4, 512, 4, 2>
              <<<dimGridPadd, dimBlockPadd>>>(ker, paddFullCuda, b);
    }

    //cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    constexpr int newInputSize = 256 * 7 * 7 * 64 * 5 * 5;
    dbl_t *newInputCuda = tensor_alloc(newInputSize);

    #pragma unroll
    for (int b = 0; b < 256; ++b)
    {
      im2col_trans<5, 5, 1600, 8, 8, 7, 7, 1, 1, 1, 256, 64>
              <<<25, 64>>>(newInputCuda, data, b);
    }

    cudaEventRecord(stopImg, 0);
    cudaEventSynchronize(stopImg);
    float timeImg;
    cudaEventElapsedTime(&timeImg, start, stopImg);

    //cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    constexpr int totalKernelSize = 7 * 7 * 64 * 512;
    dbl_t *newKernelCuda = tensor_alloc(totalKernelSize);
    dim3 dimGrid(totalKernelSize / 32);
    dim3 dimBlock(32);
    ker_transform_cuda<7, 7, 64, 512><<<dimGrid, dimBlock>>>(paddFullCuda, newKernelCuda);

    //cudaDeviceSynchronize();
    cudaEventRecord(stopKer, 0);
    cudaEventSynchronize(stopKer);
    float timeKer;
    cudaEventElapsedTime(&timeKer, start, stopKer);

    gpuErrchk(cudaPeekAtLastError());

    constexpr int resSize1 = 256 * 5 * 5 * 512;
    dbl_t *resConvCuda = tensor_alloc(resSize1);
    dim3 dimBlockConv(16, 4);
    dim3 dimGridConv(100, 32);
    back_mat_mul_cuda16<3136, 6400, 1605632, 20070400, 3276800><<<dimGridConv, dimBlockConv>>>(
                    newKernelCuda, newInputCuda, resConvCuda);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    //cudaDeviceSynchronize();

    dim3 dimGridTransf(400, 128);
    dim3 dimBlockTransf(16, 4);
    transform_res_ker<6400, 5, 5, 512, 256, 16><<<dimGridTransf, dimBlockTransf>>>(
                      resConvCuda, res);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    gpuErrchk(cudaPeekAtLastError());

    std::ofstream fos("time_gemm_dk.log", std::ios::app);
    fos << "time (dk_gemm_d3_Img) = " << timeImg << "ms\n";
    fos << "time (dk_gemm_d3_Ker) = " << timeKer << "ms\n";
    fos << "time (dk_gemm_d3_all) = " << time << "ms\n" << std::endl;

    tensor_free(newKernelCuda);
    tensor_free(newInputCuda);
    tensor_free(resConvCuda);
    tensor_free(paddFullCuda);
  }

}
