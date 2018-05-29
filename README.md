# CPU / GPU Implemention of a DCGAN in C++ / Cuda

The project is in early stages.

## Build

You need to download the celebA dataset. 
It must  be located at ./celebA

```
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


## Testing

```
make check
```
