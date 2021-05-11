import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import common

for benchmark in ["Ctwp", "Jsp", "Mosp", "Tsppd"]:
    print("Plotting {} ...".format(benchmark))

    instances_dir = "./{}/{}/json".format(benchmark, common.Directories["data"])
    instances = os.listdir(instances_dir)
    instances.sort(key=lambda f: os.stat("{}/{}".format(instances_dir,f)).st_size, reverse=True)
    instances = instances[:20]
    instances = [i.split(".")[0] for i in instances]

    results_dir = "./{}/{}".format(benchmark, common.Directories["results"])
    results_files = os.listdir(results_dir)
    dataframes = []
    for results_file in results_files:
        df = pd.read_csv("{}/{}".format(results_dir, results_file), sep=";")
        df = df[["Instance", "Timeout", "Cost", "Time"]]
        solver_name = results_file.split("-")[0]
        df["Solver"] = solver_name
        dataframes.append(df)
    df = pd.concat(dataframes, ignore_index=True)
    #df = df[df.Instance.isin(instances)]
    timeouts = df.Timeout.unique()
    for timeout in timeouts:
        df1 = df[df["Timeout"] == timeout]
        plt.figure(figsize=(0.3*df1.shape[0], 4))
        p = sns.barplot(x="Instance", y="Cost", hue="Solver", data=df1, ci=None)
        p.set_xticklabels(p.get_xticklabels(), rotation=90)
        p.set_ylabel("Cost")
        p.set_title("Timeout {}s".format(timeout))
        plot_file = "{}-{}-cost.pdf".format(benchmark, timeout)
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.clf()
        p = sns.barplot(x="Instance", y="Time", hue="Solver", data=df1, ci=None)
        p.set_xticklabels(p.get_xticklabels(), rotation=90)
        p.set_ylabel("Time [s]")
        p.set_title("Timeout {}s".format(timeout))
        plot_file = "{}-{}-time.pdf".format(benchmark, timeout)
        plt.savefig(plot_file, bbox_inches="tight")
        plt.clf()
        #plt.show()
