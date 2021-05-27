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
orders_str = str(txt_content["orders"])
orders_str = re.sub(" ", "", orders_str)
orders_str = re.sub("\[\[", "[|", orders_str)
orders_str = re.sub("\]\]", "|]", orders_str)
orders_str = re.sub("\],\[", ",\n          |", orders_str)
dzn_content_str = "c = {};\np = {};\norders = {};".format(txt_content["clients"], txt_content["products"],orders_str)

# Write dzn
dzn_filepath = os.path.splitext(txt_filepath)[0] + ".dzn"
dzn_file = open(dzn_filepath, "w")
dzn_file.write(dzn_content_str)
dzn_file.close()
