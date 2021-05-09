import argparse

# Solvers configurations
Solvers = {
    "yuck":
        {
            "path": "../Solvers/yuck/yuck.msc",
            "free_search": False
        },
    "ortools":
        {
            "path": "../Solvers/ortools/ortools.msc",
            "free_search": True
        },
    "mdd":
        {
            "path": "../../cmake-build-remote-release/mdd-gpu",
        }
}


# Directories configuration
Directories = {
    "data": "data",
    "results": "results"
}


# Auxiliary classes
class Result:
    def __init__(self, cost, search_time, solution):
        self.cost = cost
        self.search_time = search_time
        self.solution = solution

    def __iter__(self):
        for e in [self.cost, self.search_time, self.solution]:
          yield e


class BestResultWithinTimeout:
    def __init__(self, timeout):
        self.timeout = timeout
        self.best_result = None

    def reset(self):
        self.best_result = None

    def update(self, results):
        for result in results:
            if result.search_time <= self.timeout:
                if self.best_result:
                    if result.cost < self.best_result.cost:
                        self.best_result = result
                else:
                    self.best_result = result


class BenchmarksManager:
    def __init__(self, args, output_file_path):
        # Best result for each timeout
        timeouts = range(args.i, args.t + 1, args.i)
        self.best_results = [BestResultWithinTimeout(timeout) for timeout in timeouts]
        # Output file
        self.output_file = open(output_file_path, "w")
        self.output_file.write("Instance;Timeout;Cost;Time;Solution\n")
        self.output_file.flush()

    def __del__(self):
        self.output_file.close()

    def update(self, instance, results):
        # Collect best results
        for brwt in self.best_results:
            brwt.reset()
            brwt.update(results)
        # Write best results
        for brwt in self.best_results:
            self.output_file.write("{};{};".format(instance, brwt.timeout))
            if brwt.best_result:
                cost, search_time, solution = tuple(brwt.best_result)
                self.output_file.write("{};{:.3f};{}\n".format(cost, search_time, solution or ""))
            else:
                self.output_file.write(";;;\n")
            self.output_file.flush()


def parse_solver(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str, action="store", dest="s", required=True, choices=["ortools", "yuck", "mdd"])
    args, _ = parser.parse_known_args(argv)
    return args.s