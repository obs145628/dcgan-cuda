set(SRC
  activ.cc
  copy-initializer.cc
  cost.cc
  layers.cc
  normal-initializer.cc
  sgd-optimizer.cc
  zero-initializer.cc
)

add_library(api_obj OBJECT ${SRC})
