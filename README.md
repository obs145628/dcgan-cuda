# CPU / GPU Implemention of a DCGAN in C++ / Cuda

The project is in early stages.

## Build

You need to download the celebA dataset. 
It must  be located at ./celebA

```shell
git submodule update --init --recursive
cd ./ext/tocha/
mkdir _build
cd _build
cmake ..
make
cd ../../../
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
mkdir _build
cd _build
cmake ..
make
```


## Modes

To switch between the CPU/GPU execution, binaries must be run with:
```shell
RT_MODE=<mode> ./bin
```

Available modes:
- CPU: monothreaded on CPU, default value
- MCPU: multihreaded on CPU
- GPU: cuda version


## Testing

```shell
make check
```


## MNIST

Launch the MNIST classifier:

```shell
RT_MODE=<mode> ./nn_mnist mnist.data
```

Basic network with dense layers, softmax cross entropy, and adam optimizer.


## Tensorflow DCGAN

Implementation of the DCGAN with tensorflow:

```shell
cd gan_tf
python main.py
```