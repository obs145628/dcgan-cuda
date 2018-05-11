import os
import sys

import json_ts_builder
import json_ts_reader
import tensors_saver


ROOT_DIR = sys.argv[1]
BUILD_DIR = sys.argv[2]

TEST_BUILD_DIR = BUILD_DIR
TEST_DIR = os.path.join(ROOT_DIR, 'tests/')
SCRIPTS_DIR = os.path.join(TEST_DIR, 'scripts/')
ERRORS_PATH = os.path.join(BUILD_DIR, 'errors.log')

builder = json_ts_builder.JsonTsBuilder()

def test_datset_weights(cat, sub, ref_script, bin_file, dataset):

    tname = cat + '_' + sub

    builder.add_test(cat, sub,
                cmd = [
                    os.path.join(SCRIPTS_DIR, 'test_dataset_weights.sh'),
                    TEST_DIR,
                    BUILD_DIR,
                    os.path.join(TEST_DIR, ref_script),
                    os.path.join(BUILD_DIR, bin_file),
                    os.path.join(BUILD_DIR, dataset),
                    os.path.join(TEST_BUILD_DIR, tname + '_weights.npz'),
                    os.path.join(TEST_BUILD_DIR, tname + '_out_ref.npz'),
                    os.path.join(TEST_BUILD_DIR, tname + '_out_test.tbin'),
                ],
                code = 0)

def test_basic(cat, sub, ref_script, bin_file):

    tname = cat + '_' + sub

    builder.add_test(cat, sub,
                cmd = [
                    os.path.join(SCRIPTS_DIR, 'test_basic.sh'),
                    TEST_DIR,
                    BUILD_DIR,
                    os.path.join(TEST_DIR, ref_script),
                    os.path.join(BUILD_DIR, bin_file),
                    os.path.join(TEST_BUILD_DIR, tname + '_out_ref.npz'),
                    os.path.join(TEST_BUILD_DIR, tname + '_out_test.tbin'),
                ],
                code = 0)

test_datset_weights('nn', 'mnist1', 'ref_mnist1.py', 'test_mnist1', 'mnist.data')
test_datset_weights('nn', 'mnist_grad', 'ref_mnist_grad.py', 'test_mnist_grad', 'mnist.data')

test_basic('ops', 'softmax', 'ref_softmax.py', 'test_softmax')
test_basic('ops', 'log_softmax', 'ref_log_softmax.py', 'test_log_softmax')
test_basic('ops', 'softmax_cross_entropy',
           'ref_softmax_cross_entropy.py', 'test_softmax_cross_entropy')
test_basic('ops', 'conv2d', 'ref_conv2d.py', 'test_conv2d')
test_basic('ops', 'conv2d_bias_add', 'ref_conv2d_bias_add.py', 'test_conv2d_bias_add')
test_basic('ops', 'sigmoid', 'ref_sigmoid.py', 'test_sigmoid')
test_basic('ops', 'mat_mat_mul', 'ref_mat_mat_mul.py', 'test_mat_mat_mul')
test_basic('ops', 'mat_rvect_add', 'ref_mat_rvect_add.py', 'test_mat_rvect_add')
test_basic('ops', 'mean_squared_error', 'ref_mse.py', 'test_mse')
test_basic('ops', 'vect_relu', 'ref_vect_relu.py', 'test_vect_relu')
test_basic('ops', 'vect_relu_leaky', 'ref_vect_relu_leaky.py', 'test_vect_relu_leaky')
test_basic('ops', 'vect_tanh', 'ref_vect_tanh.py', 'test_vect_tanh')
test_basic('ops', 'mat_mul_add', 'ref_mat_mul_add.py', 'test_mat_mul_add')
test_basic('ops', 'tmat_mat_mul', 'ref_tmat_mat_mul.py', 'test_tmat_mat_mul')
test_basic('ops', 'mat_tmat_mul', 'ref_mat_tmat_mul.py', 'test_mat_tmat_mul')
test_basic('ops', 'mat_sum0', 'ref_mat_sum0.py', 'test_mat_sum0')
test_basic('ops', 'mat_sum1', 'ref_mat_sum1.py', 'test_mat_sum1')
test_basic('ops', 'sigmoid_cross_entropy', 'ref_sigmoid_cross_entropy.py', 'test_sigmoid_cross_entropy')

test_basic('ops_grad', 'mse_grad', 'ref_mse_grad.py', 'test_mse_grad')
test_basic('ops_grad', 'sigmoid_grad', 'ref_sigmoid_grad.py', 'test_sigmoid_grad')
test_basic('ops_grad', 'mat_mul_add_grad', 'ref_mat_mul_add_grad.py', 'test_mat_mul_add_grad')
test_basic('ops_grad', 'softmax_cross_entrop_grad', 'ref_softmax_cross_entropy_grad.py', 'test_softmax_cross_entropy_grad')
test_basic('ops_grad', 'sigmoid_cross_entropy_grad', 'ref_sigmoid_cross_entropy_grad.py', 'test_sigmoid_cross_entropy_grad')
test_basic('ops_grad', 'conv2d_grad', 'ref_conv2d_grad.py', 'test_conv2d_grad')

ts = json_ts_reader.JsonTsReader(builder.tests, True).ts
if not os.path.isfile(ERRORS_PATH):
    ts.set_run_init(True)
ts.out_err = open(ERRORS_PATH, 'w')
ts.run()
