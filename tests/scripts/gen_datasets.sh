#!/bin/sh

TESTS_DIR="$1"
BUILD_DIR="$2"
SRC_DIR="$3"

python ${TESTS_DIR}/gen_mnist.py ${BUILD_DIR}/mnist.data
python ${TESTS_DIR}/gen_celeba.py ${SRC_DIR} ${BUILD_DIR}/celeba.npz
