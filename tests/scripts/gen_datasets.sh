#!/bin/sh

TESTS_DIR="$1"
BUILD_DIR="$2"

python ${TESTS_DIR}/gen_mnist.py ${BUILD_DIR}/mnist.data
