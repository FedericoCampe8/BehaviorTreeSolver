import sys
import time
import pathlib
import asyncio
import datetime
import minizinc
import argparse
import benchmarksCommon

def getArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver",  type=str, action="store", dest="solver",  required=True, choices=["ortools", "yuck"])
    parser.add_argument("-j", "--jobs",    type=int, action="store", dest="jobs",    default=1)
    parser.add_argument("-t", "--timeout", type=int, action="store", dest="timeout", default=100000)
    args = parser.parse_args(argv)
    return args 

async def solve(args, mzn_file, dzn_file):
    # Create model
    model = minizinc.Model([mzn_file, dzn_file])
    
    # Lookup solver
    solver_config = benchmarksCommon.solvers_configs[args.solver]
    solver = minizinc.Solver.load(pathlib.Path(solver_config))
   
    # Solve
    cost = None
    search_time = None
    solution = None
    instance = minizinc.Instance(solver, model)
    timedelta_timeout = datetime.timedelta(seconds=args.timeout)
    start_time = time.perf_counter()
    async for result in instance.solutions(timeout=timedelta_timeout, processes=args.jobs, intermediate_solutions=True, free_search=True):
        if result.solution != None:
            search_time = time.perf_counter() - start_time
            cost = result.objective       
            solution = result["cfp"]
            
    return cost, search_time, solution

### Main
args = getArguments(sys.argv[1:])

# Initialize csv file
output_filename = "ctwp-" + args.solver + "-" + str(int(time.time())) + ".csv"
output_file = open(output_filename, "w")
output_file.write("Timeout:{};Jobs:{}\n".format(args.timeout, args.jobs))
output_file.write("Instance;Cost;Time;Solution\n")
output_file.flush()

# Solve instances
for instance in benchmarksCommon.instances:
    # Run solver
    print("Solving " + instance + "...")
    cost, search_time, solution = asyncio.run(solve(args, "./Mz2.mzn", "./data/dzn/" + instance + ".dzn"))
    # Write results     
    if solution:
        output_file.write("{};{};{:.3f};{}\n".format(instance, cost, search_time, solution or ""))
    else:
        output_file.write("{};;;\n".format(instance))
    output_file.flush()

# Close csv file
output_file.close()
