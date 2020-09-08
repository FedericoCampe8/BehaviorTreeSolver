Preparation:
Due to github repo limitations, before anything else go to
    lib/linux/protobuf_3_12_2
and unzip:
- libprotobuf.zip
- libprotoc.zip
REMEMBER to not push these two files.
These are too big for github.

To build:
- makedir build
- cd build
- cmake -DCMAKE_BUILD_TYPE=Debug ..
- make -j8

To run:
- ./bt_solver
