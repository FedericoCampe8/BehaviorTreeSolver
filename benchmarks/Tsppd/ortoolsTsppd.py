import sys
import time
import json
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

class solutionParser():

    def __init__(self, routing, manager):
        self.__routing = routing
        self.__manager = manager
        self.__start_time = time.perf_counter()
        self.__cost = None
        self.__search_time = None    
        self.__solution = None

    def OnSolutionCallback(self):       
        self.__search_time = time.perf_counter() - self.__start_time
        self.__cost = 0
        self.__solution = []
        vehicle_id = 0
        index = self.__routing.Start(vehicle_id)
        while not self.__routing.IsEnd(index):
            self.__solution.append(self.__manager.IndexToNode(index))
            previous_index = index
            index = self.__routing.NextVar(index).Value()
            self.__cost = self.__cost + self.__routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        self.__solution.append(self.__manager.IndexToNode(index))
        
    def getBestSolution(self):
        return self.__cost, self.__search_time, self.__solution


def initializeData(json_content):
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
    
    # Load json
    json_filepath = json_file
    json_file = open(json_filepath, "r")
    json_content = json.load(json_file)
    json_file.close()
    
    # Instantiate the data problem
    data = initializeData(json_content)

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
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        50000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100000)

    # Define Transportation Requests
    for request in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(
                delivery_index))
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index) <=
            distance_dimension.CumulVar(delivery_index))

    # Setting search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
    search_parameters.time_limit.seconds = args.timeout

    # Solve the problem
    solution_parser = solutionParser(routing, manager)
    routing.AddAtSolutionCallback(solution_parser.OnSolutionCallback)
    routing.SolveWithParameters(search_parameters)

    return solution_parser.getBestSolution()
