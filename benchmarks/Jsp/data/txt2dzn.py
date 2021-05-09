import os
import re
import sys
import common

# Parse txt
txt_filepath = sys.argv[1]
txt_file = open(txt_filepath, "r")
txt_tree = common.txt_grammar.parse(txt_file.read())
txt_file.close()
txt_content = common.TxtVisitor().visit(txt_tree)

# Fix array format
tasks_str = str(txt_content["tasks"])
tasks_str = re.sub(" ", "", tasks_str)
tasks_str = re.sub("\[\[\[", "[|", tasks_str)
tasks_str = re.sub("\]\]\]", "|]", tasks_str)
tasks_str = re.sub("\]\],", ",\n", tasks_str)
tasks_str = re.sub("\[\[", "       |", tasks_str)
tasks_str = re.sub("\],\[", ", ", tasks_str)
dzn_content_str = "n = {};\nm = {};\njob = {};".format(txt_content["jobs"], txt_content["machines"],tasks_str)

# Write dzn
dzn_filepath = os.path.splitext(txt_filepath)[0] + ".dzn"
dzn_file = open(dzn_filepath, "w")
dzn_file.write(dzn_content_str)
dzn_file.close()