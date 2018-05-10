set(SRC
  conv2d.cc
  graph.cc
  input.cc
  log-softmax.cc
  mat-mat-mul.cc
  mat-mul-add.cc
  mat-rvect-add.cc
  mat-sum.cc
  mse.cc
  mse-grad.cc
  op.cc
  ops-builder.cc
  relu-grad.cc
  shape.cc
  sigmoid-grad.cc
  softmax.cc
  softmax-cross-entropy.cc
  softmax-cross-entropy-grad.cc
  variable.cc
  vect-sigmoid.cc
  vect-relu.cc
  vect-relu-leaky.cc
  vect-tanh.cc
)

add_library(ops_obj OBJECT ${SRC})
