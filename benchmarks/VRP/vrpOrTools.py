#!/usr/bin/env python3

"""Simple Pickup Delivery Problem (PDP)."""

import time
import json
import sys

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_model(inputFile):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = inputFile["edges"]
    data['pickups_deliveries'] = [[i,i+1] for i in range(1,len(inputFile["nodes"]),2)]
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def print_solution(data, manager, routing, solution, searchTypeName, elapsedTimeMs):
    """Prints solution on console."""
    vehicle_id = 0
    index = routing.Start(vehicle_id)
    plan_output = "[RESULT] Solution: ["
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += '{},'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        last_distance = routing.GetArcCostForVehicle(
            previous_index, index, vehicle_id)
        route_distance += routing.GetArcCostForVehicle(
            previous_index, index, vehicle_id)
    plan_output += '{}]'.format(manager.IndexToNode(index))
    plan_output += ' | Value: {}'.format(route_distance)
    print(plan_output)

def main():
    """Entry point of the program."""
    grubHubInstanceFile = sys.argv[1]
    timeout = int(sys.argv[2])
    searchTypeName = sys.argv[3]

    searchTypes = {
        "GREEDY_DESCENT" : (routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT),
        "GUIDED_LOCAL_SEARCH" : (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH),
        "SIMULATED_ANNEALING" : (routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING),
        "TABU_SEARCH" : (routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
    }

    searchType = searchTypes[searchTypeName]

    grubHubInstanceJson = json.load(open(grubHubInstanceFile,"r"))
    
    """Adjust GrubHub instance"""
    del grubHubInstanceJson["nodes"][1]
    del grubHubInstanceJson["edges"][1]
    for l in grubHubInstanceJson["edges"]:
        del l[0]
    
    # Instantiate the data problem.
    data = create_model(grubHubInstanceJson)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Define cost of each arc.
    def distance_callback(from_index, to_index):
        """Returns the manhattan distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        50000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100000)

    # Define Transportation Requests.
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

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    search_parameters.time_limit.seconds = timeout

    # Solve the problem.
    search_parameters.local_search_metaheuristic = searchType
    startTime = time.time_ns() // 1000000
    solution = routing.SolveWithParameters(search_parameters)
    endTime = time.time_ns() // 1000000

    # Print solution on console
    if solution:
        print_solution(data, manager, routing, solution, searchTypeName, endTime - startTime)
    else:
        print("[RESULT] Solution: [] | Value: 0")



if __name__ == '__main__':
    main()
