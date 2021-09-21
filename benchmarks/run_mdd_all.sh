#!/usr/bin/bash

if [ "$#" -ne 1 ];
then
    echo "[ERROR]: Usage: bash $(basename $0) <Timeout>."
    exit 1
fi

timeout_sec=$1

for b in sop ctwp ossp; do
  # Compile mdd-gpu
  sed -i "24s/.*/using ProblemType = ${b^^}roblem;/" ../src/Main.cu
  sed -i "25s/.*/using StateType = ${b^^}State;/" ../src/Main.cu
  cmake --build ../cmake-build-remote-release/

  bash run_mdd_cpu-only.sh ${b} ${timeout_sec};
  bash run_mdd_cpu-gpu.sh ${b} ${timeout_sec};
done;


    
