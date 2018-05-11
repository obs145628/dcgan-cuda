set(SRC
  conv2d.cc
  graph.cc
  input.cc
  log-softmax.cc
  mat-mat-mul.cc
  mat-rvect-add.cc
  mse.cc
  op.cc
  ops-builder.cc
  reshape.cc
  shape.cc
  softmax.cc
  softmax-cross-entropy.cc
  variable.cc
  vect-sigmoid.cc
  vect-relu.cc
  vect-relu-leaky.cc
  vect-tanh.cc
)

add_library(ops_obj OBJECT ${SRC})
