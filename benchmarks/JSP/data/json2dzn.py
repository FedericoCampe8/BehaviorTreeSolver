import os
import re
import sys
import json
import numpy as np
    
# Load json
json_filepath = sys.argv[1]
json_file = open(json_filepath, "r")
json_content = json.load(json_file)
json_file.close()

# Data processing
j = json_content["jobs"]
m = json_content["machines"]
t = np.array(json_content["tasks"])
t = np.reshape(t, (j, -1))
t = t.tolist()
t = [str(job) for job in t]
t = "\n       | ".join(t)
t = re.sub("[\[\]]", "", t)

# Write dzn
dzn_filepath = os.path.splitext(json_filepath)[0] + ".dzn"
dzn_file = open(dzn_filepath, "w")
dzn_file.write("n = " + str(j) + ";\n")
dzn_file.write("m = " + str(m) + ";\n")
dzn_file.write("job = [| " + t + " |];")
dzn_file.close()
