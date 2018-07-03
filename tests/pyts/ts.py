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

TS_CPU = True
TS_MCPU = True
TS_GPU = True

builder = json_ts_builder.JsonTsBuilder()

def test_datset_weights(cat, sub, ref_script, bin_file, dataset, mode = None):

    if mode is None:
        if TS_CPU: test_datset_weights(cat, sub, ref_script, bin_file, dataset, 'CPU')
        if TS_MCPU: test_datset_weights(cat, sub, ref_script, bin_file, dataset, 'MCPU')
        if TS_GPU: test_datset_weights(cat, sub, ref_script, bin_file, dataset, 'GPU')
        return

    cat =  mode.lower() + '_' + cat
    tname = cat + '_' + sub

    builder.add_test(cat, sub,
                cmd = [
                    os.path.join(SCRIPTS_DIR, 'test_dataset_weights.sh'),
                    TEST_DIR,
                    ROOT_DIR,
                    os.path.join(TEST_DIR, ref_script),
                    os.path.join(BUILD_DIR, bin_file),
                    os.path.join(BUILD_DIR, dataset),
                    os.path.join(TEST_BUILD_DIR, tname + '_weights.npz'),
                    os.path.join(TEST_BUILD_DIR, tname + '_out_ref.npz'),
                    os.path.join(TEST_BUILD_DIR, tname + '_out_test.tbin'),
                ],
                env={
                    'RT_MODE': mode
                },
                code = 0)

def test_basic(cat, sub, ref_script, bin_file, mode = None):

    if mode is None:
        if TS_CPU: test_basic(cat, sub, ref_script, bin_file, 'CPU')
        if TS_MCPU: test_basic(cat, sub, ref_script, bin_file, 'MCPU')
        if TS_GPU: test_basic(cat, sub, ref_script, bin_file, 'GPU')
        return
    
    cat =  mode.lower() + '_' + cat
    tname = cat + '_' + sub

    builder.add_test(cat, sub,
                cmd = [
                    os.path.join(SCRIPTS_DIR, 'test_basic.sh'),
                    TEST_DIR,
                    ROOT_DIR,
                    os.path.join(TEST_DIR, ref_script),
                    os.path.join(BUILD_DIR, bin_file),
                    os.path.join(TEST_BUILD_DIR, tname + '_out_ref.npz'),
                    os.path.join(TEST_BUILD_DIR, tname + '_out_test.tbin'), 
                ],
                env={
                    'RT_MODE': mode
                },
                code = 0)

test_datset_weights('nn', 'mnist1', 'ref_mnist1.py', 'test_mnist1', 'mnist.data')
test_datset_weights('nn', 'mnist_grad', 'ref_mnist_grad.py', 'test_mnist_grad', 'mnist.data')
test_datset_weights('nn', 'dcgan_discriminator',
                   'ref_dcgan_discriminator.py', 'test_dcgan_discriminator', 'celeba.npz')

test_basic('ops', 'softmax', 'ref_softmax.py', 'test_softmax')
test_basic('ops', 'softmax_bm', 'ref_softmax_bm.py', 'test_softmax_bm')
test_basic('ops', 'log_softmax', 'ref_log_softmax.py', 'test_log_softmax')
test_basic('ops', 'log_softmax_bm', 'ref_log_softmax_bm.py', 'test_log_softmax_bm')
test_basic('ops', 'softmax_cross_entropy', 'ref_softmax_cross_entropy.py', 'test_softmax_cross_entropy')
test_basic('ops', 'softmax_cross_entropy_bm', 'ref_softmax_cross_entropy_bm.py', 'test_softmax_cross_entropy_bm')
test_basic('ops', 'conv2d', 'ref_conv2d.py', 'test_conv2d')
test_basic('ops', 'conv2d_padding', 'ref_conv2d_padding.py', 'test_conv2d_padding')
test_basic('ops', 'conv2d_bias_add', 'ref_conv2d_bias_add.py', 'test_conv2d_bias_add')
test_basic('ops', 'conv2d_transpose', 'ref_conv2d_transpose.py', 'test_conv2d_transpose')
test_basic('ops', 'sigmoid', 'ref_sigmoid.py', 'test_sigmoid')
test_basic('ops', 'sigmoid_bm', 'ref_sigmoid_bm.py', 'test_sigmoid_bm')
test_basic('ops', 'sigmoid_50k', 'ref_sigmoid_50k.py', 'test_sigmoid_50k')
test_basic('ops', 'mat_mat_mul', 'ref_mat_mat_mul.py', 'test_mat_mat_mul')
test_basic('ops', 'mat_mat_mul_bm', 'ref_mat_mat_mul_bm.py', 'test_mat_mat_mul_bm')
test_basic('ops', 'mat_rvect_add', 'ref_mat_rvect_add.py', 'test_mat_rvect_add')
test_basic('ops', 'mat_rvect_add_bm', 'ref_mat_rvect_add_bm.py', 'test_mat_rvect_add_bm')
test_basic('ops', 'mean_squared_error', 'ref_mse.py', 'test_mse')
test_basic('ops', 'mean_squared_error_bm', 'ref_mse_bm.py', 'test_mse_bm')
test_basic('ops', 'vect_relu', 'ref_vect_relu.py', 'test_vect_relu')
test_basic('ops', 'vect_relu_bm', 'ref_vect_relu_bm.py', 'test_vect_relu_bm')
test_basic('ops', 'vect_relu_leaky', 'ref_vect_relu_leaky.py', 'test_vect_relu_leaky')
test_basic('ops', 'vect_relu_leaky_bm', 'ref_vect_relu_leaky_bm.py', 'test_vect_relu_leaky_bm')
test_basic('ops', 'vect_tanh', 'ref_vect_tanh.py', 'test_vect_tanh')
test_basic('ops', 'vect_tanh_bm', 'ref_vect_tanh_bm.py', 'test_vect_tanh_bm')
test_basic('ops', 'reshape', 'ref_reshape.py', 'test_reshape')
test_basic('ops', 'mat_mul_add', 'ref_mat_mul_add.py', 'test_mat_mul_add')
test_basic('ops', 'tmat_mat_mul', 'ref_tmat_mat_mul.py', 'test_tmat_mat_mul')
test_basic('ops', 'mat_tmat_mul', 'ref_mat_tmat_mul.py', 'test_mat_tmat_mul')
test_basic('ops', 'mat_sum0', 'ref_mat_sum0.py', 'test_mat_sum0')
test_basic('ops', 'mat_sum1', 'ref_mat_sum1.py', 'test_mat_sum1')
test_basic('ops', 'sigmoid_cross_entropy', 'ref_sigmoid_cross_entropy.py', 'test_sigmoid_cross_entropy')
test_basic('ops', 'sigmoid_cross_entropy_bm', 'ref_sigmoid_cross_entropy_bm.py', 'test_sigmoid_cross_entropy_bm')
test_basic('ops', 'argmax_accuracy', 'ref_argmax.py', 'test_argmax')
test_basic('ops', 'update', 'ref_update.py', 'test_update')
test_basic('ops', 'update_bm', 'ref_update_bm.py', 'test_update_bm')
test_basic('ops', 'moment_update', 'ref_moment_update.py', 'test_moment_update')
test_basic('ops', 'moment_update_bm', 'ref_moment_update_bm.py', 'test_moment_update_bm')
test_basic('ops', 'moment_update2', 'ref_moment_update2.py', 'test_moment_update2')
test_basic('ops', 'moment_update2_bm', 'ref_moment_update2_bm.py', 'test_moment_update2_bm')
test_basic('ops', 'adam_update', 'ref_adam_update.py', 'test_adam_update')

test_basic('ops_grad', 'mse_grad', 'ref_mse_grad.py', 'test_mse_grad')
test_basic('ops_grad', 'mse_grad_bm', 'ref_mse_grad_bm.py', 'test_mse_grad_bm')
test_basic('ops_grad', 'sigmoid_grad', 'ref_sigmoid_grad.py', 'test_sigmoid_grad')
test_basic('ops_grad', 'sigmoid_grad_bm', 'ref_sigmoid_grad_bm.py', 'test_sigmoid_grad_bm')
test_basic('ops_grad', 'mat_mul_add_grad', 'ref_mat_mul_add_grad.py', 'test_mat_mul_add_grad')
test_basic('ops_grad', 'softmax_cross_entropy_grad', 'ref_softmax_cross_entropy_grad.py', 'test_softmax_cross_entropy_grad')
test_basic('ops_grad', 'softmax_cross_entropy_grad_bm', 'ref_softmax_cross_entropy_grad_bm.py', 'test_softmax_cross_entropy_grad_bm')
test_basic('ops_grad', 'sigmoid_cross_entropy_grad', 'ref_sigmoid_cross_entropy_grad.py', 'test_sigmoid_cross_entropy_grad')
test_basic('ops_grad', 'sigmoid_cross_entropy_grad_bm', 'ref_sigmoid_cross_entropy_grad_bm.py', 'test_sigmoid_cross_entropy_grad_bm')
test_basic('ops_grad', 'tanh_grad', 'ref_tanh_grad.py', 'test_tanh_grad')
test_basic('ops_grad', 'tanh_grad_bm', 'ref_tanh_grad_bm.py', 'test_tanh_grad_bm')

test_basic('ops_grad', 'conv2d_grad', 'ref_conv2d_grad.py', 'test_conv2d_grad')
test_basic('ops_grad', 'conv2d_bias_add_grad', 'ref_conv2d_bias_add_grad.py', 'test_conv2d_bias_add_grad')
test_basic('ops_grad', 'conv2d_transpose_grad', 'ref_conv2d_transpose_grad.py', 'test_conv2d_transpose_grad')
test_basic('ops_grad', 'relu_grad', 'ref_relu_grad.py', 'test_relu_grad')
test_basic('ops_grad', 'relu_grad_bm', 'ref_relu_grad_bm.py', 'test_relu_grad_bm')
test_basic('ops_grad', 'leaky_relu_grad', 'ref_leaky_relu_grad.py', 'test_leaky_relu_grad')
test_basic('ops_grad', 'conv2d_padding_grad', 'ref_conv2d_padding_grad.py', 'test_conv2d_padding_grad')


'''
test_datset_weights('discriminator', 'conv_layer0', 'ref_conv_d0.py', 'test_conv_d0', '')
test_datset_weights('discriminator', 'conv_layer1', 'ref_conv_d1.py', 'test_conv_d1', '')
test_datset_weights('discriminator', 'conv_layer2', 'ref_conv_d2.py', 'test_conv_d2', '')
test_datset_weights('discriminator', 'conv_layer3', 'ref_conv_d3.py', 'test_conv_d3', '')

test_datset_weights('discriminator', 'conv_layer0_dx', 'ref_conv_dx_d0.py', 'test_conv_dx_d0', '')
test_datset_weights('discriminator', 'conv_layer1_dx', 'ref_conv_dx_d1.py', 'test_conv_dx_d1', '')
test_datset_weights('discriminator', 'conv_layer2_dx', 'ref_conv_dx_d2.py', 'test_conv_dx_d2', '')
test_datset_weights('discriminator', 'conv_layer3_dx', 'ref_conv_dx_d3.py', 'test_conv_dx_d3', '')

test_datset_weights('discriminator', 'conv_layer0_dk', 'ref_conv_dk_d0.py', 'test_conv_dk_d0', '')
test_datset_weights('discriminator', 'conv_layer1_dk', 'ref_conv_dk_d1.py', 'test_conv_dk_d1', '')
test_datset_weights('discriminator', 'conv_layer2_dk', 'ref_conv_dk_d2.py', 'test_conv_dk_d2', '')
test_datset_weights('discriminator', 'conv_layer3_dk', 'ref_conv_dk_d3.py', 'test_conv_dk_d3', '')

test_datset_weights('generator', 'conv_layer0', 'ref_conv_g0.py', 'test_conv_g0', '')
test_datset_weights('generator', 'conv_layer1', 'ref_conv_g1.py', 'test_conv_g1', '')
test_datset_weights('generator', 'conv_layer2', 'ref_conv_g2.py', 'test_conv_g2', '')
test_datset_weights('generator', 'conv_layer3', 'ref_conv_g3.py', 'test_conv_g3', '')

test_datset_weights('generator', 'conv_layer0_dx', 'ref_conv_dx_g0.py', 'test_conv_dx_g0', '')
test_datset_weights('generator', 'conv_layer1_dx', 'ref_conv_dx_g1.py', 'test_conv_dx_g1', '')
test_datset_weights('generator', 'conv_layer2_dx', 'ref_conv_dx_g2.py', 'test_conv_dx_g2', '')
test_datset_weights('generator', 'conv_layer3_dx', 'ref_conv_dx_g3.py', 'test_conv_dx_g3', '')

test_datset_weights('generator', 'conv_layer0_dk', 'ref_conv_dk_g0.py', 'test_conv_dk_g0', '')
test_datset_weights('generator', 'conv_layer1_dk', 'ref_conv_dk_g1.py', 'test_conv_dk_g1', '')
test_datset_weights('generator', 'conv_layer2_dk', 'ref_conv_dk_g2.py', 'test_conv_dk_g2', '')
test_datset_weights('generator', 'conv_layer3_dk', 'ref_conv_dk_g3.py', 'test_conv_dk_g3', '')
'''

ts = json_ts_reader.JsonTsReader(builder.tests, True).ts
if not os.path.isfile(ERRORS_PATH):
    ts.set_run_init(True)
ts.out_err = open(ERRORS_PATH, 'w')
ts.run()