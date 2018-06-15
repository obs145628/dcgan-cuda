#include <iostream>
#include <cmath>

#include <tocha/tensor.hh>
#include "../src/memory/alloc.hh"

template<int widthA, int widthB, int blockDimX>
__global__
void mat_mul_cuda(const dbl_t *A, const dbl_t *B, dbl_t *C)
{
  __shared__ dbl_t A_tile[blockDimX * blockDimX];
  dbl_t cRes[blockDimX] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  const int aStart = widthA * blockDimX * blockIdx.y;
  const int aEnd = aStart + widthA - 1;
  const int bStart = blockDimX * 4 * blockIdx.x;
  const int bStep = blockDimX * widthB;

  int cIdx = widthB * blockDimX * blockIdx.y + blockDimX * blockIdx.x * 4
                  + blockDimX * threadIdx.y + threadIdx.x;

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
    const dbl_t *bPartial = &B[bPIndex];
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

  #pragma unroll
  for (int i = 0; i < blockDimX; ++i)
  {
    C[cIdx] = cRes[i];
    cIdx += widthB;
  }
}

void ker_transformed_print(const dbl_t *ker, int nbFilter, int size)
{
  std::cout << "[\n";
  for (int filter = 0; filter < nbFilter; ++filter)
  {
    for(int i = 0; i < size; ++i)
      std::cout << ker[filter * size + i] << ", ";
    std::cout << "\n";
  }
  std::cout << "\n]";
}

__global__
void ker_transform_cuda(const dbl_t *ker, dbl_t *res, const int hSize,
                        const int wSize, const int chSize,
                        const int nbFilter)
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

void print_patch(const int *patch, const int width, const int height, const int P,
                const int Q)
{
  std::cout << std::endl;
  for (int i = 0; i < width * height; ++i)
  {
    std::cout << "(" << (i%width) << ", " << (i/width) << ") = [";
    for (int j = 0; j < P * Q - 1; ++j)
      std::cout << patch[i * P * Q + j] << ", ";
    std::cout << patch[i * P * Q + P * Q - 1] << "], ";
  }
  std::cout << std::endl;
}

__global__
void mark_patch(int *patch, const int stride, const int wSize, const int P,
                const int Q,
                const int wkSize)
{
  const int top_left = blockIdx.y * stride * wSize + blockIdx.x * stride;
  const int patchIdx = blockIdx.y * Q + blockIdx.x;
  const int relativeIdx = threadIdx.y * wkSize + threadIdx.x;
  const int fullIdx = top_left + threadIdx.y * wSize + threadIdx.x;
  patch[fullIdx * P * Q + patchIdx] = relativeIdx;
}


void img_transformed_print(const dbl_t *img, const int batch, const int chSize,
                           const int P, const int Q, const int wkSize, const int hkSize)
{
  std::cout << std::endl;
  const int patchSize = P * Q * batch;
  for (int i = 0; i < chSize * wkSize * hkSize; ++i)
  {
    for(int j = 0; j < patchSize; ++j)
      std::cout << img[i * patchSize + j] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template<int wSize, int hSize, int chSize, int wkSize, int hkSize, int P, int Q,
        int batchSize, int padTop, int padBot, int padL, int padR, int stot0, int stot1>
__global__
void img_transform_cuda(const dbl_t *img, dbl_t *res, const int *patchIndex)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int batchIdx = index / stot0;
  const int chIdx = (index - batchIdx * stot0) % chSize;
  const int hIdx = (index - batchIdx * stot0 - chIdx) / (wSize * chSize);
  const int wIdx = (index - batchIdx * stot0 - chIdx - hIdx * (wSize * chSize)) / chSize;
  int i = 0;
  const int *patchPtr = &patchIndex[hIdx * wSize * stot1
                         + wIdx * stot1];
  int patch;

  dbl_t val = 0;

  if (wIdx >= padL && wIdx < (wSize - padR)
      && hIdx >= padTop && hIdx < (hSize - padBot))
  {

    const int oldIndex = batchIdx * 64 * 64 * 3 + (hIdx - padTop) * 64 * 3
            + (wIdx - padL) * 3 + chIdx;

    val = img[oldIndex];

    #pragma unroll
    for (;i < stot1; i++)
    {
      patch = *(patchPtr + i);
      if (patch != -1)
      {
        int nindex = chIdx * P * Q * wkSize * hkSize * batchSize
                     + patch * P * Q * batchSize
                     + batchIdx * P * Q
                     + i;
        res[nindex] = val;
      }
    }
  }
}

template<int wSize, int hSize, int chSize, int wkSize, int hkSize, int P, int Q,
        int batchSize, int padTop, int padBot, int padL, int padR, int stot0, int stot1>
__global__
void img_transform_cuda2(const dbl_t *img, dbl_t *res, const int *patchIndex)
{
  const int index = blockIdx.y * 13467 * 4 + blockIdx.x * 201
                    + threadIdx.y * 13467 + threadIdx.x;
  const int batchIdx = index / stot0;
  const int chIdx = (index - batchIdx * stot0) % chSize;
  const int hIdx = (index - batchIdx * stot0 - chIdx) / (wSize * chSize);
  const int wIdx = (index - batchIdx * stot0 - chIdx - hIdx * (wSize * chSize)) / chSize;
  int i = 0;
  const int *patchPtr = &patchIndex[hIdx * wSize * stot1
                         + wIdx * stot1];
  int patch;

  dbl_t val = 0;

  if (wIdx >= padL && wIdx < (wSize - padR)
      && hIdx >= padTop && hIdx < (hSize - padBot))
  {

    const int oldIndex = batchIdx * 64 * 64 * 3 + (hIdx - padTop) * 64 * 3
            + (wIdx - padL) * 3 + chIdx;

    val = img[oldIndex];

    #pragma unroll
    for (;i < stot1; i++)
    {
      patch = *(patchPtr + i);
      if (patch != -1)
      {
        int nindex = chIdx * P * Q * wkSize * hkSize * batchSize
                     + patch * P * Q * batchSize
                     + batchIdx * P * Q
                     + i;
        res[nindex] = val;
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
  const int P = (64 + 3 - heightKer) / stride + 1;
  const int Q = (64 + 3 - widthKer) / stride + 1;
  const int newInputSize = chSize * heightKer * widthKer * batchSize * P * Q;

  float milli = 0;

  dbl_t *kernelCuda, *newKernelCuda;

  cudaEvent_t start, stop, startConv, stopKer, startPatch, stopPatch, startImg, stopImg;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&startConv);
  cudaEventCreate(&startPatch);
  cudaEventCreate(&startImg);
  cudaEventCreate(&stopKer);
  cudaEventCreate(&stopPatch);
  cudaEventCreate(&stopImg);

  cudaEventRecord(start);
  cudaMalloc((void**)&kernelCuda, sizeof(dbl_t) * totalKernelSize);
  cudaMalloc((void**)&newKernelCuda, sizeof(dbl_t) * totalKernelSize);
  cudaMemcpy(kernelCuda, kernelHost, sizeof(dbl_t) * totalKernelSize,
                  cudaMemcpyHostToDevice);
  dim3 dimGrid(totalKernelSize / 32);
  dim3 dimBlock(32);
  ker_transform_cuda<<<dimGrid, dimBlock>>>(kernelCuda, newKernelCuda,
                  heightKer, widthKer, chSize, nbFilter);
  cudaEventRecord(stopKer);
  //cudaDeviceSynchronize();
  cudaEventSynchronize(stopKer);

  cudaEventElapsedTime(&milli, start, stopKer);
  std::cout << "Timer Ker: " << milli << " ms" << std::endl;

  cudaEventRecord(startPatch);
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
  cudaEventRecord(stopPatch);
  //cudaDeviceSynchronize();
  cudaEventSynchronize(stopPatch);

  cudaEventElapsedTime(&milli, startPatch, stopPatch);
  std::cout << "Timer Patch: " << milli << " ms" << std::endl;

  dbl_t *zeroInit = (dbl_t*)calloc(newInputSize, sizeof(dbl_t));
  cudaMemcpy(newInputCuda, zeroInit, sizeof(dbl_t) * newInputSize, cudaMemcpyHostToDevice);

  /*const int stotImg = width * height * chSize;
  dim3 dimGridImg(totalInputSize / 32);
  dim3 dimBlockImg(32);
  cudaEventRecord(startImg);
  img_transform_cuda<67, 67, 3, 5, 5, 32, 32, 64, 1, 2, 1, 2, 13467, 1024>
          <<<dimGridImg, dimBlockImg>>>(inputCuda, newInputCuda, patchCuda);*/
  dim3 dimGridImg(67, 16);
  dim3 dimBlockImg(201, 4);
  cudaEventRecord(startImg);
  img_transform_cuda2<67, 67, 3, 5, 5, 32, 32, 64, 1, 2, 1, 2, 13467, 1024>
          <<<dimGridImg, dimBlockImg>>>(inputCuda, newInputCuda, patchCuda);
  cudaEventRecord(stopImg);
  //cudaDeviceSynchronize();
  cudaEventSynchronize(stopImg);

  cudaEventElapsedTime(&milli, startImg, stopImg);
  std::cout << "Timer Img: " << milli << " ms" << std::endl;

  dbl_t *resConvCuda;
  const int resSize = nbFilter * batchSize * P * Q;
  cudaMalloc((void**)&resConvCuda, sizeof(dbl_t) * resSize);
  dim3 dimBlockConv(16, 4);
  dim3 dimGridConv(1024, 4);

  cudaEventRecord(startConv);
  mat_mul_cuda<75, 65536, 16><<<dimGridConv, dimBlockConv>>>(newKernelCuda, newInputCuda, resConvCuda);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&milli, startConv, stop);
  std::cout << "Timer conv: " << milli << " ms" << std::endl;
  cudaEventElapsedTime(&milli, start, stop);
  std::cout << "Timer all: " << milli << " ms" << std::endl;

  dbl_t *transfCuda;
  cudaMalloc((void**)&transfCuda, sizeof(dbl_t) * resSize);
  dim3 dimGridTransf(4096, 16);
  dim3 dimBlockTransf(16, 4);
  transform_res<<<dimGridTransf, dimBlockTransf>>>(resConvCuda, transfCuda);

  cudaDeviceSynchronize();

  tocha::Tensors out;
  out.add(tocha::Tensor::f32(64, 32, 32, 64));
  dbl_t* res = reinterpret_cast<dbl_t*>(out.arr()[0].data);

  cudaMemcpy(res, transfCuda, sizeof(dbl_t) * resSize, cudaMemcpyDeviceToHost);

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
