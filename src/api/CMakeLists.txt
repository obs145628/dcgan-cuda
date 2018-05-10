set(SRC
  activ.cc
  copy-initializer.cc
  cost.cc
  layers.cc
  normal-initializer.cc
  zero-initializer.cc
)

add_library(api_obj OBJECT ${SRC})
