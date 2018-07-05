# CPU / GPU Implemention of a DCGAN in C++ / Cuda

Implementation of a DCGAN to generates faces, trained with the celebA dataset.

## Build

You need to download the celebA dataset. (next section)

```shell
./bootstrap
cd _build
make
```


## celebA dataset

You need to dowload the celebA dataset and to put it at ./celebA.
Then run:
```shell
cd _build
python ../dcgan/preprocess.py
```
This command scales all the dataset, it might takes some time.

## DCGAN
```shell
cd _build
RT_MODE=<mode> ./dcgan
```

Options: 
--model <model-file>: load a pretrained model if the file exists, and save the model during training 
--train <epochs>: train the network for several epochs 
--generate <jpg-out-path>: generate a sample of faces and save it to a jpg file 


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
cd _build
make check
```


## MNIST

Launch the MNIST classifier:

```shell
cd _build
RT_MODE=<mode> ./nn_mnist mnist.data
```

Basic network with dense layers, softmax cross entropy, and adam optimizer.