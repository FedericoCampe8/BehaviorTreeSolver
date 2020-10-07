#!/bin/bash

function printHelp 
{
    echo "Usage: ./autoCMake.sh <Platform type> <Build type> [--remote]"
    echo "  Platform types:"
    echo "    -c,--cpu      CPU"
    echo "    -g,--gpu      GPU"
    echo "  Build types:"
    echo "    -d,--debug    Debug"
    echo "    -r,--release  Release"
    echo "Examples:"
    echo "  ./autoCMake.sh --cpu --release"
    echo "  ./autoCMake.sh -g -d --remote"
}


if [ "$#" -lt 2 ];
then
    printHelp
    exit 1
fi

platformType=""
buildType=""
hostType="Local"

for arg in "$@"
do
    case "$arg" in
        -c|--cpu)
            platformType="CPU"
            ;;
        -g|--gpu)
            platformType="GPU"
            ;;
        -r|--release)
            buildType="Release"
            ;;
        -d|--debug)
            buildType="Debug"
            ;;

        --remote)
            hostType="Remote"
            ;;
        -h|--help)
            printHelp
            exit 0
            ;;
        *)
            echo "[ERROR] Unrecognized argument: $arg"
            printHelp
            exit 1
            ;;
    esac
done

if [ -z "$platformType" ];
then
    echo "[ERROR] Missing platform type"
    printHelp
    exit 1
fi

if [ -z "$buildType" ];
then
    echo "[ERROR] Missing build type"
    printHelp
    exit 1
fi

buildDir="build"
if [ "$hostType" == "Remote" ];
then
  buildDir="build-remote"
fi

rm -rf ./"$buildDir" &> /dev/null
mkdir ./"$buildDir"
cd ./"$buildDir"

cmake \
    -DPLATFORM_TYPE=$platformType \
    -DCMAKE_BUILD_TYPE=$buildType \
    ..
