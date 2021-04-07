#!/usr/bin/env python3

import sys
import subprocess
import time
import benchmarksCommon

outFileName = "VrpBenchmarksOrTools-"+ str(int(time.time())) + ".csv"
outFile = open(outFileName, "w")

for grubHubInstance in benchmarksCommon.grubHubInstances:
    for timeout in benchmarksCommon.timeouts:
        for searchTypeName in benchmarksCommon.searchTypeNames:
            cmd = \
                ["/usr/bin/python3"] + \
                ["./vrpOrTools.py"] + \
                [grubHubInstance] + \
                [str(timeout)] + \
                [searchTypeName]

            print("Running " + str(cmd) + " -> ", end = "")
            sys.stdout.flush()
            output = "[RESULT] Solution: [] | Value: 0"
            try:
                tmpOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="ascii")
                output = tmpOutput.split("\n")[-2]
            except subprocess.CalledProcessError:
               pass
            value = benchmarksCommon.getSolutionValue(output)
            print(value)
            info = value + ", " + "{0}".format(', '.join(map(str, cmd[1:])))
            outFile.write(info + "\n")
            outFile.flush()

outFile.close()