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
    def __init__(self, cost, solution):
        self.cost = cost
        self.solution = solution

    def __iter__(self):
        for e in [self.cost, self.solution]:
          yield e

class BenchmarksManager:
    def __init__(self, args, output_file_path):
        # Output file
        self.output_file = open(output_file_path, "w")
        self.output_file.write("Instance;Cost;Solution\n")
        self.output_file.flush()

    def __del__(self):
        self.output_file.close()

    def update(self, instance, results):
        if results:
            cost, solution = tuple(results[-1])
            self.output_file.write("{};{};{}\n".format(instance, cost, solution))
        else:
            self.output_file.write("{};;\n".format(instance))
        self.output_file.flush()


def parse_solver(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str, action="store", dest="s", required=True, choices=["ortools", "yuck", "mdd"])
    args, _ = parser.parse_known_args(argv)
    return args.s