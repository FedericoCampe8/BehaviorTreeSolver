#!/usr/bin/env python3

import sys
import subprocess
import os

def getSolutionValue(output):
    rawFields = [f.strip() for f in output.split('|')]
    return rawFields[1].split(' ')[1]

### Main ###

# Main queue reduced sizes
mainQueueReducedSizes = []
mainQueueReducedSizeMin = 1000
mainQueueReducedSizeMax = 1024000
mainQueueReducedSize = mainQueueReducedSizeMin
while mainQueueReducedSize <= mainQueueReducedSizeMax:
    mainQueueReducedSizes.append(mainQueueReducedSize)
    mainQueueReducedSize = mainQueueReducedSize * 2

# Main queue max sizes
mainQueueMaxSizes = [] 
mainQueueMaxSizeMin = 1024000
mainQueueMaxSizeMax = 16384000
mainQueueMaxSize = mainQueueMaxSizeMin
while mainQueueMaxSize <= mainQueueMaxSizeMax:
    mainQueueMaxSizes.append(mainQueueMaxSize)
    mainQueueMaxSize = mainQueueMaxSize * 2

gpuParallelisms = [1000,2000,4000]

binPath = str(sys.argv[1])
inFilePath = str(sys.argv[2])
timeoutSeconds = str(sys.argv[3])
mddWidth = str(sys.argv[4])

inFileName = os.path.basename(inFilePath).split('.')[0]
outFileName = inFileName + "T" + timeoutSeconds + "W" + mddWidth + ".csv"
outFile = open(outFileName, "w")

for mainQueueMaxSize in mainQueueMaxSizes:
    for mainQueueReducedSize in mainQueueReducedSizes:
        for gpuParallelism in gpuParallelisms:
            cmd = [binPath] + [inFilePath] + [timeoutSeconds] + [str(mainQueueMaxSize)] + [str(mainQueueReducedSize)] + [mddWidth] + [str(gpuParallelism)]
            print("Running " + str(cmd) + " -> ", end="" )
            sys.stdout.flush()
            output = "[RESULT] Solution: [] | Value: 0 | Time: 0 ms (00h00m00s) | Iterations: 0 | Visited states: 0"
            try:
                tmpOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="ascii")
                output = tmpOutput.split("\n")[-2]
            except subprocess.CalledProcessError:
               pass
            value = getSolutionValue(output)
            print(value)
            info = value + "," + str(mainQueueMaxSize) + "," + str(mainQueueReducedSize) + "," + str(gpuParallelism)
            outFile.write(info + "\n")

outFile.close()