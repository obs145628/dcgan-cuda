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
test_basic('ops', 'softmax', 'ref_softmax.py', 'test_softmax')
test_basic('ops', 'log_softmax', 'ref_log_softmax.py', 'test_log_softmax')
test_basic('ops', 'softmax_cross_entropy',
           'ref_softmax_cross_entropy.py', 'test_softmax_cross_entropy')
test_basic('ops', 'conv2d', 'ref_conv2d.py', 'test_conv2d')
test_basic('ops', 'sigmoid', 'ref_sigmoid.py', 'test_sigmoid')
test_basic('ops', 'mat_mat_mul', 'ref_mat_mat_mul.py', 'test_mat_mat_mul')
test_basic('ops', 'mat_rvect_add', 'ref_mat_rvect_add.py', 'test_mat_rvect_add')
test_basic('ops', 'mean_squared_error', 'ref_mse.py', 'test_mse')





ts = json_ts_reader.JsonTsReader(builder.tests, True).ts
if not os.path.isfile(ERRORS_PATH):
    ts.set_run_init(True)
ts.out_err = open(ERRORS_PATH, 'w')
ts.run()
