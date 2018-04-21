#!/bin/sh

TESTS_DIR="$1"
BUILD_DIR="$2"


python ${TESTS_DIR}/gen_mnist.py ${BUILD_DIR}/mnist.data
python ${TESTS_DIR}/ref_mnist.py ${BUILD_DIR}/weights.npz ${BUILD_DIR}/mnist_ref.npz
${BUILD_DIR}/mnist_test ${BUILD_DIR}/mnist.data ${BUILD_DIR}/weights.npz ${BUILD_DIR}/mnist_out.tbin
${BUILD_DIR}/bin/tbin-dump ${BUILD_DIR}/mnist_ref.npz
${BUILD_DIR}/bin/tbin-dump ${BUILD_DIR}/mnist_out.tbin
${BUILD_DIR}/bin/tbin-diff ${BUILD_DIR}/mnist_ref.npz ${BUILD_DIR}/mnist_out.tbin 
