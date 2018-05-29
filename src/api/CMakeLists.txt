set(SRC
  activ.cu
  adam-optimizer.cu
  copy-initializer.cu
  cost.cu
  layers.cu
  normal-initializer.cu
  sgd-optimizer.cu
  zero-initializer.cu
)

add_library(api_obj OBJECT ${SRC})
