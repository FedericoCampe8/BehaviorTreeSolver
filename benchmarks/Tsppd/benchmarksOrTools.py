import sys
import time
import argparse
import benchmarksCommon
import ortoolsTsppd

def getArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timeout",       type=int, action="store", dest="timeout",       default=100000)
    args = parser.parse_args(argv)
    return args 

### Main
args = getArguments(sys.argv[1:])

# Initialize csv file
output_filename = "tsppd-{}-{}-{}-{}.csv".format("ortools", args.timeout, 1, int(time.time()))
output_file = open(output_filename, "w")
output_file.write("Timeout:{};\n".format(args.timeout))
output_file.write("Instance;Cost;Time;Solution\n")
output_file.flush()

for instance in benchmarksCommon.instances:
    # Run solver
    print("Solving " + instance + "...")
    cost, search_time, solution = ortoolsTsppd.solve(args, "./data/json/" + instance + ".json")    
    # Write results
    if solution:
        output_file.write("{};{};{:.3f};{}\n".format(instance, cost, search_time, solution or ""))
    else:
        output_file.write("{};;;\n".format(instance))
    output_file.flush()

output_file.close()
