import time
import pathlib
import datetime
import minizinc
import argparse
from common import Result, Solvers


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str, action="store", dest="s", required=True, choices=["ortools", "yuck"])
    parser.add_argument("-t", type=int, action="store", dest="t", default=60)
    parser.add_argument("-i", type=int, action="store", dest="i", default=15)
    parser.add_argument("-j", type=int, action="store", dest="j", default=1)
    args, _ = parser.parse_known_args(argv)
    return args


def get_args_str(args):
    return "t{}-i{}-j{}-{}".format(args.t, args.i, args.j, int(time.time()))


async def solve(args, mzn_file, dzn_file, solution_variable):
    # Create model
    model = minizinc.Model([mzn_file, dzn_file])
    # Lookup solver
    solver_config = Solvers[args.s]
    solver = minizinc.Solver.load(pathlib.Path(solver_config["path"]))
    # Solve
    results = []
    instance = minizinc.Instance(solver, model)
    timedelta_timeout = datetime.timedelta(seconds=args.t)
    start_time = time.perf_counter()
    async for result in instance.solutions(timeout=timedelta_timeout, processes=args.j, intermediate_solutions=True, free_search=solver_config["free_search"]):
        if result.solution:
            search_time = time.perf_counter() - start_time
            cost = result.objective
            solution = result[solution_variable]
            results.append(Result(cost, search_time, solution))
    return results
