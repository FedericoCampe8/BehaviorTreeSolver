import sys
import asyncio
from instances import Instances

sys.path.append("..")
import common
import commonMdd
import commonMinizinc

### Main
argv = sys.argv[1:]
solver = common.parse_solver(argv)
args = None
args_str = ""
if solver == "mdd":
    args = commonMdd.parse_args(argv)
    args_str = commonMdd.get_args_str(args)
else:
    args = commonMinizinc.parse_args(argv)
    args_str = commonMinizinc.get_args_str(args)
output_file_path = "./{}/{}-{}.csv".format(common.Directories["results"], solver, args_str)
benchmarks_manager = common.BenchmarksManager(args, output_file_path)

for instance in Instances:
    print("Solving {} ...".format(instance))
    results = []
    if solver == "mdd":
        json_file = "./{}/json/{}.json".format(common.Directories["data"], instance)
        results = commonMdd.solve(args, json_file)
    else:
        mzn_file = "./jobshop.mzn"
        dzn_file = "./{}/dzn/{}.dzn".format(common.Directories["data"], instance)
        results = asyncio.run(commonMinizinc.solve(args, mzn_file, dzn_file, "ts"))
    benchmarks_manager.update(instance, results)
