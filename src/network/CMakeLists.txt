set(SRC
  activation.cc
  cost-function.cc
  fully-connected-layer.cc
  layer.cc
  network.cc
)

add_library(network_obj OBJECT ${SRC})
