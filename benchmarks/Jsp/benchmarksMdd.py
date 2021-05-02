import sys
import time
import argparse
import benchmarksCommon
import mddJsp

def getArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-q",    type=int, action="store", dest="q")
    parser.add_argument("-t",    type=int, action="store", dest="t")
    parser.add_argument("--wc",  type=int, action="store", dest="wc")
    parser.add_argument("--pc",  type=int, action="store", dest="pc")
    parser.add_argument("--wg",  type=int, action="store", dest="wg")
    parser.add_argument("--pg",  type=int, action="store", dest="pg")
    parser.add_argument("--eq",  type=int, action="store", dest="eq")
    parser.add_argument("--neq", type=int, action="store", dest="neq")
    args = parser.parse_args(argv)
    return args 

### Main
args = getArguments(sys.argv[1:])
flat_args = args.t, args.q, args.wc, args.pc, args.wg, args.pg, args.eq, args.neq

# Initialize csv file
output_filename = "jsp-{}-t{}-q{}-wc{}-pc{}-wg{}-pg{}-eq{}-neq{}-{}.csv".format("mdd", *flat_args, int(time.time()))
output_file = open(output_filename, "w")
output_file.write("Timeout:{};Queue_size:{};Width_CPU:{};Parallelism_CPU:{};Width_GPU:{};Parallelism_CPU:{};Lns_Percentage_Eq:{};Lns_Percentage_Neq:{} \n".format(*flat_args))
output_file.write("Instance;Cost;Time;Solution\n")
output_file.flush()

for instance in benchmarksCommon.instances:
    # Run solver
    print("Solving " + instance + "...")
    cost, search_time, solution = mddJsp.solve(args, "./data/json/" + instance + ".json")    
    # Write results     
    if solution:
        output_file.write("{};{};{:.3f};{}\n".format(instance, cost, search_time, solution or ""))
    else:
        output_file.write("{};;;\n".format(instance))
    output_file.flush()

output_file.close()
