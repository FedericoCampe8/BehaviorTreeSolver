import sys
import ortoolsJsp
from instances import Instances

sys.path.append("..")
import common
import commonMdd

### Main
argv = sys.argv[1:]
solver = common.parse_solver(argv)
args = None
args_str = ""
if solver == "mdd":
    args = commonMdd.parse_args(argv)
    args_str = commonMdd.get_args_str(args)
else:
    args = ortoolsJsp.parse_args(argv)
    args_str = ortoolsJsp.get_args_str(args)
output_file_path = "./{}/{}-{}.csv".format(common.Directories["results"], solver, args_str)
benchmarks_manager = common.BenchmarksManager(args, output_file_path)

for instance in Instances:
    print("Solving {} ...".format(instance))
    results = []
    json_file = "./{}/json/{}.json".format(common.Directories["data"], instance)
    if solver == "mdd":
        results = commonMdd.solve(args, json_file)
    else:
        results = ortoolsJsp.solve(args, json_file)
    benchmarks_manager.update(instance, results)
