import os
import numpy as np
import pandas as pd
import common

good_cost_threshold_percentage = 115
good_time_threshold_percentage = 75

def incrementValue(df, statistic, timeout, solver):
    row_index = df.index[(df["Statistic"] == statistic) & (df["Timeout"] == timeout)].tolist()[0]
    df.at[row_index, solver] += 1

for benchmark in ["Ctwp", "Jsp", "Mosp", "Tsppd"]:
    print("Summarizing {} ...".format(benchmark))
    results_dir = "./{}/{}".format(benchmark, common.Directories["results"])
    results_files = os.listdir(results_dir)
    results_files.sort()
    dataframes = []
    for results_file in results_files:
        df = pd.read_csv("{}/{}".format(results_dir, results_file), sep=";")
        df = df[["Instance", "Timeout", "Cost", "Time"]]
        solver_name = results_file.split("-")[0]
        df["Solver"] = solver_name
        dataframes.append(df)
    data = pd.concat(dataframes, ignore_index=True)
    # Calculate minimums
    minimums = data[["Instance", "Timeout", "Cost", "Time"]]
    minimums = minimums.sort_values(by=["Instance", "Timeout", "Cost", "Time"], ignore_index=True, na_position="last")
    minimums = minimums.drop_duplicates(subset=["Instance", "Timeout"], ignore_index=True, keep="first")
    # Create empty summary
    solvers = sorted(data.Solver.unique())
    statistics = ["Best solution", "Good solution in less time", "Unsolved"]
    timeouts = sorted(data.Timeout.unique())
    summary_columns = ["Statistic", "Timeout"] + solvers
    summary = pd.DataFrame(columns=summary_columns)
    for statistic in statistics:
        for timeout in timeouts:      
            summary.loc[len(summary)] = [statistic, timeout] + [0] * len(solvers)
    # Fill summary
    instances = data.Instance.unique()
    for timeout in timeouts:
        for instance in instances:
            for solver in solvers:
                row = data[(data["Instance"] == instance) & (data["Timeout"] == timeout) & (data["Solver"] == solver)]
                cost = row["Cost"]
                time = row["Time"]
                if not cost.dropna().empty:
                    cost_value = row["Cost"].iat[0]
                    time_value = row["Time"].iat[0]
                    min_row = minimums[(minimums["Instance"] == instance) & (minimums["Timeout"] == timeout)]
                    min_cost_value = min_row["Cost"].iat[0]
                    min_time_value = min_row["Time"].iat[0]                    
                    if cost_value <= min_cost_value and time_value <= min_time_value:
                        incrementValue(summary, "Best solution", timeout, solver) 
                    elif cost_value <= good_cost_threshold_percentage / 100 * min_cost_value and time_value <= good_time_threshold_percentage / 100 * max(1,min_time_value):
                        incrementValue(summary, "Good solution in less time", timeout, solver) 
                else:
                   incrementValue(summary, "Unsolved", timeout, solver) 
    summary.sort_values(["Timeout", "Statistic"], inplace=True)
    summary.rename(columns={"mdd":"LNS-MDD", "ortools":"OR-Tools", "yuck":"Yuck"}, inplace=True)
    # Latex output
    #print(summary.to_string(index = False))
    print(summary.to_latex(index = False))
