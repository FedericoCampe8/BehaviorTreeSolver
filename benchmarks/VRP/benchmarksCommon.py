def getSolutionValue(output):
    rawFields = [f.strip() for f in output.split('|')]
    return rawFields[1].split(' ')[1]

grubHubInstances = {
#    "./data/grubhub-06-0.json" : 4613,
#    "./data/grubhub-07-0.json" : 5401,
#    "./data/grubhub-08-0.json" : 7394,
#    "./data/grubhub-09-0.json" : 7607,
#    "./data/grubhub-10-0.json" : 7881,
    "./data/grubhub-15-0.json" : 10072,
    "./data/grubhub-15-1.json" : 8437,
    "./data/grubhub-15-2.json" : 9510,
    "./data/grubhub-15-3.json" : 10414,
    "./data/grubhub-15-4.json" : 10035,
    "./data/grubhub-15-5.json" : 10580,
    "./data/grubhub-15-6.json" : 8693,
    "./data/grubhub-15-7.json" : 9961,
    "./data/grubhub-15-8.json" : 9959,
    "./data/grubhub-15-9.json" : 11721
}
timeouts = [1,10,30,60,300]
searchTypeNames = [
    "GREEDY_DESCENT",
    "GUIDED_LOCAL_SEARCH",
    "SIMULATED_ANNEALING",
    "TABU_SEARCH"
]