import os
import numpy as np
import pandas as pd
import common

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
    statistics = ["Avg % cost diff", "Avg time diff [s]", "Unsolved"]
    timeouts = sorted(data.Timeout.unique())
    summary_columns = ["Solver", "Statistic", "Timeout", "Value"]
    summary = pd.DataFrame(columns=summary_columns)
    # Fill summary
    instances = data.Instance.unique()
    for solver in solvers:
        for timeout in timeouts:
            avg_cost_diff = 0.0
            avg_time_diff = 0.0
            solved_instances = 0.0
            unsolved_instances = 0.0
            for instance in instances:
                row = data[(data["Instance"] == instance) & (data["Timeout"] == timeout) & (data["Solver"] == solver)]
                cost = row["Cost"]
                time = row["Time"]
                if (pd.isna(cost)).bool():
                    unsolved_instances += 1
                else:
                    cost = row["Cost"].iat[0]
                    time = row["Time"].iat[0]
                    min_row = minimums[(minimums["Instance"] == instance) & (minimums["Timeout"] == timeout)]
                    min_cost = min_row["Cost"].iat[0]
                    min_time = min_row["Time"].iat[0]
                    solved_instances += 1
                    avg_cost_diff += (cost - min_cost) / (1 if min_cost <= 1 else min_cost)
                    avg_time_diff += time - min_time
            avg_cost_diff = 100 * avg_cost_diff / solved_instances
            avg_time_diff = avg_time_diff / solved_instances
            summary.loc[len(summary)] = [solver, statistics[0], timeout, avg_cost_diff]
            summary.loc[len(summary)] = [solver, statistics[1], timeout, avg_time_diff]
            summary.loc[len(summary)] = [solver, statistics[2], timeout, unsolved_instances]

    summary = summary.sort_values(by=["Statistic", "Timeout", "Solver", "Value"], ignore_index=True)
    print(summary)