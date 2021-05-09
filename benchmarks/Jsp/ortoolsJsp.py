import sys
import time
import json
import argparse
import collections
from ortools.sat.python import cp_model

sys.path.append("..")
import common

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=int, action="store", dest="t", default=60)
    parser.add_argument("-i", type=int, action="store", dest="i", default=15)
    parser.add_argument("-j", type=int, action="store", dest="j", default=1)
    args, _ = parser.parse_known_args(argv)
    return args


def get_args_str(args):
    return "t{}-i{}-j{}-{}".format(args.t, args.i, args.j, int(time.time()))


class SolutionParser(cp_model.CpSolverSolutionCallback):
    def __init__(self, cost_var, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.cost_var = cost_var
        self.variables = variables
        self.start_time = time.perf_counter()
        self.results = []

    def OnSolutionCallback(self): 
        cost = self.Value(self.cost_var)
        search_time = time.perf_counter() - self.start_time
        solution = []
        for v in self.variables:
            solution.append(self.Value(v))
        self.results.append(common.Result(cost, search_time, solution))

    def get_results(self):
        return self.results


def solve(args, json_file):
    # Create the model
    model = cp_model.CpModel()

    # Load json
    json_filepath = json_file
    json_file = open(json_filepath, "r")
    json_content = json.load(json_file)
    json_file.close()

    # Data processing
    j = json_content["jobs"]
    m = json_content["machines"]
    jobs_data = json_content["tasks"]

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Named tuple to store information about created variables
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    start_times = []
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            start_times.append(start_var)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Solve model
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = args.j
    solver.parameters.max_time_in_seconds = args.t
    
    solution_parser = SolutionParser(obj_var, start_times)
    solver.SolveWithSolutionCallback(model, solution_parser)
    
    return solution_parser.get_results()
