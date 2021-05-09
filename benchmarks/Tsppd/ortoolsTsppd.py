import sys
import time
import json
import argparse
import threading
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

sys.path.append("..")
import common

metaheuristics = [
    (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH),
    (routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING),
    (routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
]

def jobs(j):
    jobs = int(j)
    if jobs % len(metaheuristics) != 0:
        raise argparse.ArgumentTypeError("Only multiple of {} jobs allowed".format(len(metaheuristics)))
    return jobs

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=int, action="store", dest="t", default=60)
    parser.add_argument("-i", type=int, action="store", dest="i", default=15)
    parser.add_argument("-j", type=jobs, action="store", dest="j", default=3)
    args, _ = parser.parse_known_args(argv)
    return args

def get_args_str(args):
    return "t{}-i{}-j{}-{}".format(args.t, args.i, args.j, int(time.time()))

class SolutionParser:
    def __init__(self, routing, manager):
        self.routing = routing
        self.manager = manager
        self.start_time = time.perf_counter()
        self.results = []

    def OnSolutionCallback(self):
        cost = 0
        search_time = time.perf_counter() - self.start_time
        solution = []
        vehicle_id = 0
        index = self.routing.Start(vehicle_id)
        while not self.routing.IsEnd(index):
            solution.append(self.manager.IndexToNode(index))
            previous_index = index
            index = self.routing.NextVar(index).Value()
            cost = cost + self.routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        solution.append(self.manager.IndexToNode(index))
        self.results.append(common.Result(cost, search_time, solution))
        
    def get_results(self):
        return self.results


def prepare_data(json_file_path):
    # Read json
    json_file = open(json_file_path, "r")
    json_content = json.load(json_file)
    json_file.close()
    
    # Adjust instance
    del json_content["nodes"][1]
    del json_content["edges"][1]
    for e in json_content["edges"]:
        del e[0]

    # Create data
    data = {}
    data['distance_matrix'] = json_content["edges"]
    data['pickups_deliveries'] = [[i,i+1] for i in range(1,len(json_content["nodes"]),2)]
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def solve(args, json_file):
    # Prepare input
    data = prepare_data(json_file)

    # Search
    threads = []
    threads_results = []
    for _ in range(args.j):
        r = []
        t = threading.Thread(target=solve_single_thread, args=(args, data, r, len(threads)))
        threads.append(t)
        threads_results.append(r)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results = []
    for thread_results in threads_results:
        for result in thread_results:
            results.append(result)

    results.sort(key=lambda result: result.search_time)
    return results


def solve_single_thread(args, data, results, thread_id):

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Define cost of each arc
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint
    dimension_name = 'Distance'
    routing.AddDimension(transit_callback_index, 0, 50000, True, dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100000)

    # Define Transportation Requests
    for request in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
        routing.solver().Add(distance_dimension.CumulVar(pickup_index) <= distance_dimension.CumulVar(delivery_index))

    # Setting search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    search_parameters.local_search_metaheuristic = metaheuristics[thread_id % len(metaheuristics)]
    search_parameters.time_limit.seconds = args.t

    # Solve the problem
    solution_parser = SolutionParser(routing, manager)
    routing.AddAtSolutionCallback(solution_parser.OnSolutionCallback)
    routing.SolveWithParameters(search_parameters)

    results = solution_parser.get_results()
