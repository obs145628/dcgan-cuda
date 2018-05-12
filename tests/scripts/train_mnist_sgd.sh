#!/bin/sh


W1="/tmp/w1.npz"
W2="/tmp/w2.npz"
W2_BIN="/tmp/w2.tbin"
REF="../tests/ref_train_mnist_sgd.py"
BIN="./test_train_mnist_sgd"
DATASET="mnist.data"

make test_train_mnist_sgd
python ${REF} ${W1} ${W2}
${BIN} ${DATASET} ${W1} ${W2_BIN}
./bin/tbin-diff ${W2} ${W2_BIN}
cp ${W2} ${W1}
