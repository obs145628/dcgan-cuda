set(SRC
  kernels.cc
  runner.cc
)

add_library(cpu_obj OBJECT ${SRC})
