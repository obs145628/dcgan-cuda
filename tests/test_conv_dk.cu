#include <iostream>
#include <cmath>

#include <tocha/tensor.hh>
#include "../src/memory/alloc.hh"

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
    const int dataOffset = z * (batchSize * width * height)
            + (yOldIdx * batchSize * width)
            + xOldIdx * batchSize + batchIdx;
    data += dataOffset;

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
          if (indexPartial < maxPartial && (blockIdx.x != 1 || threadIdx.y == 0 || (i + threadIdx.x < 11)))
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

   //const int blockOver = blockIdx.x * 64 + threadIdx.x;

   if (blockIdx.x == 1 && threadIdx.x >= 11)
     return;

   #pragma unroll
   for (int i = 0; i < 16; ++i)
   {
     if (blockIdx.x == 1 && threadIdx.y > 0 && (i + threadIdx.x >= 11))
        break;
     if (cIdx == 0)//481
      printf("cIdx == 0, blY=%d, blX=%d, thY=%d, thX=%d, i=%d, val=%f\n", blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, i, cRes[i]);
     if (cIdx == 4799)//481
      printf("cIdx == 4799, blY=%d, blX=%d, thY=%d, thX=%d, i=%d, val=%f\n", blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, i, cRes[i]);
     if (cIdx == 481)
      printf("cIdx == 481, blY=%d, blX=%d, thY=%d, thX=%d, i=%d, val=%f\n", blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, i, cRes[i]);
     if (cIdx == 679)
      printf("cIdx == 679, blY=%d, blX=%d, thY=%d, thX=%d, i=%d, val=%f\n", blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, i, cRes[i]);
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
     if (cIdx == 0)//481
      printf("cIdx == 0, blY=%d, blX=%d, thY=%d, thX=%d, i=%d, val=%f\n", blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, i, cRes[i]);
     if (cIdx == 4799)//481
      printf("cIdx == 4799, blY=%d, blX=%d, thY=%d, thX=%d, i=%d, val=%f\n", blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, i, cRes[i]);
     if (cIdx == 481)
      printf("cIdx == 481, blY=%d, blX=%d, thY=%d, thX=%d, i=%d, val=%f\n", blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, i, cRes[i]);
     if (cIdx == 679)
      printf("cIdx == 679, blY=%d, blX=%d, thY=%d, thX=%d, i=%d, val=%f\n", blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, i, cRes[i]);
     if (cIdx < maxC)
       C[cIdx] = cRes[i];
     cIdx += widthB;
   }
}

void make_conv(char **argv)
{
  auto mats = tocha::Tensors::load(argv[1]);
  const dbl_t* inputHost = reinterpret_cast<dbl_t*>(mats.arr()[1].data);
  const dbl_t* kernelHost = reinterpret_cast<dbl_t*>(mats.arr()[0].data);

  dbl_t *kerCuda;
  cudaMalloc((void**)&kerCuda, sizeof(dbl_t) * 64 * 32 * 32 * 64);
  cudaMemcpy(kerCuda, kernelHost, sizeof(dbl_t) * 64 * 32 * 32 * 64, cudaMemcpyHostToDevice);

  dbl_t *paddFullCuda;
  constexpr int paddFullSize = 63 * 63 * 64 * 64;
  cudaMalloc((void**)&paddFullCuda, sizeof(dbl_t) * paddFullSize);
  dbl_t *zeroVal = (dbl_t*)calloc(paddFullSize, sizeof(dbl_t));
  cudaMemcpy(paddFullCuda, zeroVal, sizeof(dbl_t) * paddFullSize, cudaMemcpyHostToDevice);
  free(zeroVal);
  dim3 dimGridPadd(128, 8);
  dim3 dimBlockPadd(16, 4);

  #pragma unroll
  for(int b = 0; b < 64; ++b)
  {
    padd_ker_rot<63*64*64, 64*64, 32, 64, 32, 2>
            <<<dimGridPadd, dimBlockPadd>>>(kerCuda, paddFullCuda, b);
  }

  cudaDeviceSynchronize();

  dbl_t *inputCuda;
  cudaMalloc((void**)&inputCuda, sizeof(dbl_t) * 64 * 64 * 64 * 3);
  cudaMemcpy(inputCuda, inputHost, sizeof(dbl_t) * 64 * 64 * 64 * 3, cudaMemcpyHostToDevice);

  dbl_t *newInputCuda;
  constexpr int newInputSize = 64 * 63 * 63 * 3 * 5 * 5;
  cudaMalloc((void**)&newInputCuda, sizeof(dbl_t) * newInputSize);

  #pragma unroll
  for (int b = 0; b < 3; ++b)
  {
    im2col_trans<5, 5, 1600, 64, 64, 63, 63, 1, 1, 1, 3, 64>
            <<<25, 64>>>(newInputCuda, inputCuda, b);
  }

  cudaDeviceSynchronize();

  dbl_t *newKernelCuda;
  constexpr int totalKernelSize = 63 * 63 * 64 * 64;
  cudaMalloc((void**)&newKernelCuda, sizeof(dbl_t) * totalKernelSize);
  dim3 dimGrid(totalKernelSize / 64);
  dim3 dimBlock(64);
  ker_transform_cuda<63, 63, 64, 64><<<dimGrid, dimBlock>>>(paddFullCuda, newKernelCuda);

  cudaDeviceSynchronize();

  dbl_t *resConvCuda;
  constexpr int resSize1 = 64 * 5 * 5 * 3;
  cudaMalloc((void**)&resConvCuda, sizeof(dbl_t) * resSize1);
  dim3 dimBlockConv(16, 4);
  dim3 dimGridConv(2, 4);
  back_mat_mul_cuda75<254016, 75, 16257024, 19051200, 4800><<<dimGridConv, dimBlockConv>>>(
                  newKernelCuda, newInputCuda, resConvCuda);

  cudaDeviceSynchronize();

  tocha::Tensors out;
  out.add(tocha::Tensor::f32(63, 63, 64, 64));
  out.add(tocha::Tensor::f32(64*63*63, 5*5*3));
  out.add(tocha::Tensor::f32(64, 63*63*64));
  out.add(tocha::Tensor::f32(64, 5*5*3));
  dbl_t* resker = reinterpret_cast<dbl_t*>(out.arr()[0].data);
  dbl_t* resinput = reinterpret_cast<dbl_t*>(out.arr()[1].data);
  dbl_t* nresker = reinterpret_cast<dbl_t*>(out.arr()[2].data);
  dbl_t* resconv = reinterpret_cast<dbl_t*>(out.arr()[3].data);

  cudaMemcpy(resker, paddFullCuda, sizeof(dbl_t) * paddFullSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(resinput, newInputCuda, sizeof(dbl_t) * newInputSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(nresker, newKernelCuda, sizeof(dbl_t) * totalKernelSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(resconv, resConvCuda, sizeof(dbl_t) * resSize1, cudaMemcpyDeviceToHost);

  printf("Rand val[481] = %f\n", resconv[481]);
  printf("val[4799] = %f\n", resconv[4799]);

  out.save(argv[2]);
}


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid number of arguments\n";
        return 1;
    }

    make_conv(argv);

    return 0;
}
