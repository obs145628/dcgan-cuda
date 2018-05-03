# CPU / GPU Implemention of a DCGAN in C++ / Cuda

The project is in early stages.

## Build

```
git submodule update --init --recursive
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
