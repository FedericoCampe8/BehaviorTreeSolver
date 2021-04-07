#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np
import benchmarksCommon
import csv
from cycler import cycler

def parseMddCsv(mddCvsName, grubHubInstance, timeouts):
    results = {}
    with open(mddCvsName, "r") as mddCsv:
        csvReader = csv.reader(mddCsv)
        for row in csvReader:
            if row[19].strip() == grubHubInstance:
                results[row[4].strip()] = row[0].strip()
    flatResults = []
    for timeout in timeouts:
        flatResults.append(int(results[str(timeout)]))

    return flatResults

def parseOrToolCsv(orToolsCvsName, grubHubInstance, timeouts, searchTypeName):
    results = {}
    with open(orToolsCvsName, "r") as orToolsCsv:
        csvReader = csv.reader(orToolsCsv)
        for row in csvReader:
            if row[2].strip() == grubHubInstance and row[4].strip() == searchTypeName:
                results[row[3].strip()] = row[0].strip()

    flatResults = []
    for timeout in timeouts:
        flatResults.append(int(results[str(timeout)]))

    return flatResults

### Main ####

mddCvsName = sys.argv[1]
orToolsCvsName = sys.argv[2]

for grubHubInstance in benchmarksCommon.grubHubInstances:
    mddResults = parseMddCsv(mddCvsName, grubHubInstance, benchmarksCommon.timeouts)
    orToolResults = {}
    for searchTypeName in benchmarksCommon.searchTypeNames:
        orToolResults[searchTypeName] = parseOrToolCsv(orToolsCvsName, grubHubInstance, benchmarksCommon.timeouts, searchTypeName)

    # Initialize plot context
    fig, ax = plt.subplots()
    colors = ['C0','C1','C2','C4','C8','C9']

    # Bars
    xs = np.arange(len(benchmarksCommon.timeouts))
    barsTotalWidth = 0.8
    barsCount = 1 + len(orToolResults)
    barWidth = barsTotalWidth / barsCount
    i = 0
    ax.bar([x - barsTotalWidth / 2 + barWidth * i for x in xs], mddResults, barWidth, align='edge', label='MDD', color=colors[i])
    i = i + 1
    for searchTypeName in orToolResults:
        ax.bar([x - barsTotalWidth / 2 + barWidth * i for x in xs], orToolResults[searchTypeName], barWidth, align='edge', label=searchTypeName,color=colors[i])
        i = i + 1

    # Optimal value
    optimalValue = benchmarksCommon.grubHubInstances[grubHubInstance]
    plt.axhline(y=optimalValue, label="Optimal value ({})".format(optimalValue), color='red', linestyle='--')

    # Labels, title, etc.
    ax.set_xlabel('Timeout')
    ax.set_ylabel('Best solution')
    ax.set_title('Benchmarks for ' + grubHubInstance)
    ax.set_xticks(xs)
    ax.set_xticklabels(benchmarksCommon.timeouts)
    plt.legend(loc='lower right')

    fig.tight_layout()

    benchmarksPlotName = "benchmarks-" + grubHubInstance.split('/')[-1].split('.')[0] + ".png"
    plt.savefig(benchmarksPlotName, bbox_inches='tight')
    plt.show()

