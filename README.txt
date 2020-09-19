## Requirements
- CMake >= 3.16
- GCC >= 9.3
- TBB >= 2020.1

### Addional requirements for GPU version
- CUDA SDK >= 10.1
- nVIDIA GPU with [compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) >= 3.5

## Compilation
Execute the following commands
```sh
./autoCMake.sh -c -d
cd build
make
```
for a **C**PU **d**ebug version, or
```sh
./autoCMake.sh -g -r
cd build
make
```
for a **G**PU **r**elease version.

Fore complete details about build configurations:
```sh
./autoCMake.sh --help
```
