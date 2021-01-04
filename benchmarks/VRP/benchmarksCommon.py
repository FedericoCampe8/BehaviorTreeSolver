def getSolutionValue(output):
    rawFields = [f.strip() for f in output.split('|')]
    return rawFields[1].split(' ')[1]

grubHubInstances = {
#    "./GrubHub/grubhub-05-0.json" : 4259,
#    "./GrubHub/grubhub-06-0.json" : 4613,
#    "./GrubHub/grubhub-07-0.json" : 5401,
#    "./GrubHub/grubhub-08-0.json" : 7394,
#    "./GrubHub/grubhub-09-0.json" : 7607,
#    "./GrubHub/grubhub-10-0.json" : 7881,
#    "./GrubHub/grubhub-11-0.json" : 8637,
#    "./GrubHub/grubhub-12-0.json" : 8622,
#    "./GrubHub/grubhub-13-0.json" : 9285,
#    "./GrubHub/grubhub-14-0.json" : 9464,
    "./GrubHub/grubhub-15-0.json" : 10072,
    "./GrubHub/grubhub-15-1.json" : 8437,
    "./GrubHub/grubhub-15-2.json" : 9510,
    "./GrubHub/grubhub-15-3.json" : 10414,
    "./GrubHub/grubhub-15-4.json" : 10035,
    "./GrubHub/grubhub-15-5.json" : 10580,
    "./GrubHub/grubhub-15-6.json" : 8693,
    "./GrubHub/grubhub-15-7.json" : 9961,
    "./GrubHub/grubhub-15-8.json" : 9959,
    "./GrubHub/grubhub-15-9.json" : 11721
}
timeouts = [1,10,30,60,300]
searchTypeNames = [
    "GREEDY_DESCENT",
    "GUIDED_LOCAL_SEARCH",
    "SIMULATED_ANNEALING",
    "TABU_SEARCH"
]