#include <iostream>
#include <cmath>

#include <tocha/tensor.hh>
#include "../src/memory/alloc.hh"

template<int widthA, int widthB, int blockDimX>
__global__
void mat_mul_cuda(dbl_t *A, dbl_t *B, dbl_t *C)
{
  __shared__ dbl_t A_tile[blockDimX * blockDimX];
  dbl_t cRes[blockDimX] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  const int aStart = widthA * blockDimX * blockIdx.y;
  const int aEnd = aStart + widthA - 1;
  const int bStart = blockDimX * 4 * blockIdx.x;
  const int bStep = blockDimX * widthB;

  for (int aIdx = aStart, bIdx = bStart;
       aIdx <= aEnd;
       aIdx += blockDimX, bIdx += bStep)
  {

    #pragma unroll
    for (int i = 0; i < 16; i += 4)
    {
      const int aIndex = aIdx + widthA * (i + threadIdx.y) + threadIdx.x;
      if (aIndex < 4800)
        A_tile[i + threadIdx.y + blockDimX * threadIdx.x] = A[aIndex];
    }

    __syncthreads();

    const int bPIndex = bIdx + blockDimX * threadIdx.y + threadIdx.x;
    dbl_t *bPartial = &B[bPIndex];
    int indexPartial = 0;
    int tileIndex = 0;

    #pragma unroll
    for (int i = 0; i < blockDimX; ++i)
    {
      if (bPIndex + indexPartial < 4915200)
      {
        const dbl_t bVal = bPartial[indexPartial];

        #pragma unroll
        for (int j = 0; j < 16; ++j)
          cRes[j] += A_tile[tileIndex + j] * bVal;
        tileIndex += blockDimX;
        indexPartial += widthB;
      }
    }

    __syncthreads();
  }

  int cIdx = widthB * blockDimX * blockIdx.y + blockDimX * blockIdx.x * 4
                  + blockDimX * threadIdx.y + threadIdx.x;
   #pragma unroll
  for (int i = 0; i < blockDimX; ++i)
  {
    C[cIdx] = cRes[i];
    cIdx += widthB;
  }
}

__global__
void ker_transform_cuda(dbl_t *ker, dbl_t *res, int hSize, int wSize, int chSize,
                        int nbFilter)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int filterIdx = index % nbFilter;
  int chIdx = ((index - filterIdx) % (chSize * nbFilter))/nbFilter;
  int wIdx = ((index - filterIdx - (chIdx * nbFilter))
                  % (chSize * nbFilter * wSize))/(chSize * nbFilter);
  int hIdx = (index - wIdx * chSize * nbFilter - chIdx * nbFilter - filterIdx)/(wSize * chSize * nbFilter);
  int nIndex = filterIdx * hSize * wSize * chSize + chIdx * wSize * hSize + hIdx * wSize + wIdx;
  res[nIndex] = ker[index];
}

__global__
void mark_patch(int *patch, const int stride, const int wSize, const int P,
                const int Q,
                const int wkSize)
{
  const int top_left = blockIdx.y * stride * wSize + blockIdx.x * stride;
  const int patchIdx = blockIdx.y * P + blockIdx.x;
  const int relativeIdx = threadIdx.y * wkSize + threadIdx.x;
  const int fullIdx = top_left + threadIdx.y * wkSize + threadIdx.x;//wSize
  patch[fullIdx * P * Q + patchIdx] = relativeIdx;
}

__global__
void img_transform_cuda(const dbl_t *img, dbl_t *res, const int wSize,
                        const int chSize, const int stot0, const int *patchIndex,
                        const int wkSize, const int hkSize, const int P,
                        const int Q, const int batchSize, const int padTop,
                        const int padBot, const int padL, const int padR)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int batchIdx = index / stot0;
  const int chIdx = (index - batchIdx * stot0) % chSize;
  const int hIdx = ((index - batchIdx * stot0 - chIdx) / chSize) / wSize;
  const int wIdx = ((index - batchIdx * stot0 - chIdx) / chSize) % wSize;
  const int stot1 = P * Q;
  int i = 0;
  const int *patchPtr = &patchIndex[hIdx * wSize * stot1
                         + wIdx * stot1];
  int patch = *patchPtr;
  if (wIdx > padL && wIdx < padR && hIdx > padTop && hIdx < padBot)
  {
    dbl_t val = img[index];

    while (i < stot1)
    {
      if (patch != -1)
      {
        int nindex = chIdx * P * Q * wkSize * hkSize * batchSize
                     + patch * P * Q * batchSize
                     + batchIdx * P * Q
                     + i;
        res[nindex] = val;
      }
      i++;
      patch = *(patchPtr + i);
    }
  }
}

void make_conv(char **argv)
{
  auto mats = tocha::Tensors::load(argv[1]);
  const dbl_t* inputHost = reinterpret_cast<dbl_t*>(mats.arr()[0].data);
  const dbl_t* kernelHost = reinterpret_cast<dbl_t*>(mats.arr()[1].data);

  const int batchSize = 64;
  const int height = 67; // 64 + 3 pad
  const int width = 67; // 64 + 3 pad
  const int chSize = 3;
  const int totalInputSize = batchSize * height * width * chSize;
  const int realTotalInputSize = batchSize * 64 * 64 * chSize;

  const int heightKer = 5;
  const int widthKer = 5;
  const int nbFilter = 64;
  const int totalKernelSize = heightKer * widthKer * chSize * nbFilter;

  const int stride = 2;
  const int padTop = 1;
  const int padBottom = 2;
  const int padL = 1;
  const int padR = 2;
  const int P = (height + 3 - heightKer) / stride + 1;
  const int Q = (width + 3 - widthKer) / stride + 1;
  const int newInputSize = chSize * heightKer * widthKer * batchSize * P * Q;

  dbl_t *kernelCuda, *newKernelCuda;

  cudaEvent_t start, stop, startConv;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&startConv);

  cudaEventRecord(start);
  cudaMalloc((void**)&kernelCuda, sizeof(dbl_t) * totalKernelSize);
  cudaMalloc((void**)&newKernelCuda, sizeof(dbl_t) * totalKernelSize);
  cudaMemcpy(kernelCuda, kernelHost, sizeof(dbl_t) * totalKernelSize,
                  cudaMemcpyHostToDevice);
  dim3 dimGrid(totalKernelSize / 32);
  dim3 dimBlock(32);
  ker_transform_cuda<<<dimGrid, dimBlock>>>(kernelCuda, newKernelCuda,
                  heightKer, widthKer, chSize, nbFilter);

  cudaDeviceSynchronize();

  dbl_t *inputCuda, *newInputCuda;
  cudaMalloc((void**)&inputCuda, sizeof(dbl_t) * realTotalInputSize);
  cudaMalloc((void**)&newInputCuda, sizeof(dbl_t) * newInputSize);
  cudaMemcpy(inputCuda, inputHost, sizeof(dbl_t) * realTotalInputSize,
                  cudaMemcpyHostToDevice);
  int *patchCuda;
  cudaMalloc((void**)&patchCuda, sizeof(int) * width * height * P * Q);
  cudaMemset(patchCuda, -1, sizeof(int) * width * height * P * Q);
  dim3 dimGridPatch(P, Q);
  dim3 dimBlockPatch(widthKer, heightKer);
  mark_patch<<<dimGridPatch, dimBlockPatch>>>(patchCuda, stride, width, P, Q, widthKer);

  cudaDeviceSynchronize();

  const int stotImg = width * height * chSize;
  dim3 dimGridImg(totalInputSize / 32);
  dim3 dimBlockImg(32);
  img_transform_cuda<<<dimGridImg, dimBlockImg>>>(inputCuda, newInputCuda,
                            width, chSize, stotImg,
                            patchCuda, widthKer, heightKer, P, Q, batchSize,
                            padTop, padBottom, padL, padR);
  cudaDeviceSynchronize();

  dbl_t *resConvCuda;
  const int resSize = nbFilter * batchSize * P * Q;
  cudaMalloc((void**)&resConvCuda, sizeof(dbl_t) * resSize);
  dim3 dimBlockConv(16, 4);
  dim3 dimGridConv(1024, 4);


  cudaEventRecord(startConv);
  mat_mul_cuda<75, 65536, 16><<<dimGridConv, dimBlockConv>>>(newKernelCuda, newInputCuda, resConvCuda);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milli = 0;
  cudaEventElapsedTime(&milli, start, stop);
  std::cout << "Timer all: " << milli << " ms" << std::endl;
  cudaEventElapsedTime(&milli, startConv, stop);
  std::cout << "Timer conv: " << milli << " ms" << std::endl;


  tocha::Tensors out;
  out.add(tocha::Tensor::f32(64, 69696));
  dbl_t* res = reinterpret_cast<dbl_t*>(out.arr()[0].data);

  cudaMemcpy(res, resConvCuda, sizeof(dbl_t) * resSize, cudaMemcpyDeviceToHost);

  out.save(argv[2]);

  cudaFree(newInputCuda);
  cudaFree(newKernelCuda);
  cudaFree(resConvCuda);
  cudaFree(inputCuda);
  cudaFree(patchCuda);
  cudaFree(kernelCuda);
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
