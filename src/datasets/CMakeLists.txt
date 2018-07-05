set(SRC
  celeba.cu
  jpg.cu
  mnist.cu
)

add_library(datasets_obj OBJECT ${SRC})
