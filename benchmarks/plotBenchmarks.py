import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import common

for benchmark in ["Ctwp", "Jsp", "Mosp", "Tsppd"]:
    print("Plotting {} ...".format(benchmark))
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
    df = pd.concat(dataframes, ignore_index=True)
    timeouts = df.Timeout.unique()
    h = 4
    w = 0.3*df.shape[0]/len(timeouts)
    g = sns.FacetGrid(df, row="Timeout", height=h, aspect=w/h, row_order=timeouts )
    # Costs
    g.map_dataframe(sns.barplot, x="Instance", y="Cost", hue="Solver", order=None, data=df, ci=None, palette="tab10")
    g.set_ylabels("Cost")
    for axes in g.axes.flat:
        axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
    g.add_legend()
    plot_file = "{}-cost.pdf".format(benchmark,)
    g.savefig(plot_file, dpi=300, bbox_inches="tight")
    # Time
    g = sns.FacetGrid(df, row="Timeout", height=h, aspect=w/h, row_order=timeouts )
    g.map_dataframe(sns.barplot, x="Instance", y="Time", hue="Solver", order=None, data=df, ci=None, palette="tab10")
    g.set_ylabels("Time [s]")
    for axes in g.axes.flat:
        axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
    g.add_legend()
    plot_file = "{}-time.pdf".format(benchmark,)
    g.savefig(plot_file, dpi=300, bbox_inches="tight")
