import sys
import time
import subprocess
import benchmarksCommon

output_filename = "JspOrTools-"+ str(int(time.time())) + ".csv"
output_file = open(output_filename, "w")
output_file.write("Instance;Cost;Time;Solution\n")
output_file.flush()

for instance in benchmarksCommon.instances:
    # Run solver
    cmd = \
        ["/usr/bin/python3"] + \
        ["./or-tools.py"] + \
        ["-j", "16"] +\
        ["-t", str(benchmarksCommon.timeout)] + \
        ["./data/json/" + instance + ".json"]
    print("Running " + " ".join(cmd))
    sys.stdout.flush()    
    output = "=========="
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="ascii")
    except subprocess.CalledProcessError:
        pass   

    # Parse output
    output_tree = benchmarksCommon.OutputGrammar.parse(output)
    cost, time, solution = benchmarksCommon.OutputVisitor().visit(output_tree)

    # Write results     
    output_file.write("{};{};{};{}\n".format(instance, cost, time, solution))
    output_file.flush()

output_file.close()
