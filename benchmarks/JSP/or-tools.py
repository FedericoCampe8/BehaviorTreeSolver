import sys
import time
import json
import argparse
import collections
import numpy as np
from ortools.sat.python import cp_model

def getArguments(argv):
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-j", "--jobs",    type=int, action="store", dest="jobs",    default=1)
    args_parser.add_argument("-t", "--timeout", type=int, action="store", dest="timeout", default=100000)
    args_parser.add_argument("json_file")
    args = args_parser.parse_args(argv)
    return args 


class solutionPrinter(cp_model.CpSolverSolutionCallback):

    def __init__(self, cost, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__cost = cost
        self.__start_time = time.perf_counter()
        self.__variables = variables

    def OnSolutionCallback(self):    
        cost = self.Value(self.__cost)   
        solution = []
        search_time = time.perf_counter() - self.__start_time
        for v in self.__variables:            
            solution.append(self.Value(v))
        print("Cost: {} | Time: {:.3f} | Solution: {} ".format(cost, search_time, solution))
        print("----------")

def solveJobshop(jobs,timeout,json_file):
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
    t = np.array(json_content["tasks"])
    t = np.reshape(t, (j,m,-1))
    jobs_data = t.tolist()

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
    solver.parameters.num_search_workers = jobs
    solver.parameters.max_time_in_seconds = timeout
    
    solution_printer = solutionPrinter(obj_var, start_times)
    solver.SolveWithSolutionCallback(model,solution_printer)
    print("==========")

# Main
args = getArguments(sys.argv[1:])
solveJobshop(args.jobs, args.timeout, args.json_file) 
