## Requirements

### CPU
An X86_64 CPU and the following software:
- CMake >= 3.16
- GCC >= 7
- TBB >= 2020.1

### GPU 
An NVIDIA GPU with [compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) >= 5.2 and the following software:
- CMake >= 3.16
- CUDA 11
- GCC 9
- TBB >= 2020.1

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
