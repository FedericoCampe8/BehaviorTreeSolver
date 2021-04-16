import sys
import time
import argparse
import benchmarksCommon
import ortoolsJsp

def getArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs",    type=int, action="store", dest="jobs",    default=1)
    parser.add_argument("-t", "--timeout", type=int, action="store", dest="timeout", default=100000)
    args = parser.parse_args(argv)
    return args 

### Main
args = getArguments(sys.argv[1:])

# Initialize csv file
output_filename = "jsp-ortools-"+ str(int(time.time())) + ".csv"
output_file = open(output_filename, "w")
output_file.write("Timeout:{};Jobs:{}\n".format(args.timeout, args.jobs))
output_file.write("Instance;Cost;Time;Solution\n")
output_file.flush()

for instance in benchmarksCommon.instances:
    # Run solver
    print("Solving " + instance + "...")
    cost, search_time, solution = ortoolsJsp.solve(args, "./data/json/" + instance + ".json")    
    # Write results     
    output_file.write("{};{};{:.3f};{}\n".format(instance, cost, search_time, solution))
    output_file.flush()

output_file.close()
