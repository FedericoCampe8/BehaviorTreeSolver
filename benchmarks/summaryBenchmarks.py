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
            tmp_columns = ["cost_diffs", "time_diffs"]
            tmp_df = pd.DataFrame(columns=tmp_columns)
            for instance in instances:
                row = data[(data["Instance"] == instance) & (data["Timeout"] == timeout) & (data["Solver"] == solver)]
                cost = row["Cost"]
                time = row["Time"]
                if not cost.dropna().empty:
                    cost = row["Cost"].iat[0]
                    time = row["Time"].iat[0]
                    min_row = minimums[(minimums["Instance"] == instance) & (minimums["Timeout"] == timeout)]
                    min_cost = min_row["Cost"].iat[0]
                    min_time = min_row["Time"].iat[0]
                    cost_diff = 100 * (cost - min_cost) / (1 if min_cost <= 1 else min_cost)
                    time_diff = time - min_time
                    tmp_df.loc[len(tmp_df)] = [cost_diff, time_diff]
            solved_instances = tmp_df.shape[0]
            unsolved_instances = len(instances) - solved_instances
            avg_cost_diff = tmp_df["cost_diffs"].sum() / tmp_df.shape[0]
            avg_time_diff = tmp_df["time_diffs"].sum() / tmp_df.shape[0]
            summary.loc[len(summary)] = [solver, statistics[0], timeout, avg_cost_diff]
            summary.loc[len(summary)] = [solver, statistics[1], timeout, avg_time_diff]
            summary.loc[len(summary)] = [solver, statistics[2], timeout, unsolved_instances]
    # Pretty print summary
    summary_formatted_columns = ["Statistic", "Timeout"] + [str(s) for s in solvers]
    summary_formatted = pd.DataFrame(columns=summary_formatted_columns)
    for timeout in timeouts:
        for statistic in statistics:
            row_data = []
            for solver in solvers:     
                summary_row = summary[(summary["Solver"] == solver) & (summary["Statistic"] == statistic) & (summary["Timeout"] == timeout)]
                row_data.append(summary_row["Value"].iat[0])
            summary_formatted.loc[len(summary_formatted)] = [statistic, timeout] + row_data
    summary_formatted.to_csv("{}-summary.csv".format(benchmark), sep=';', index=False, encoding="utf-8",float_format="%.2f")
    #print(summary_formatted)
