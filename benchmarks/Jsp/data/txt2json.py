import os
import re
import sys
import json
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# Txt grammar
txt_grammar = Grammar("""
    txt = (comment nl)* int ws+ int ws+ tasks
    comment = "#" ~"."+
    tasks = (ws* int ws*)*    
    int = ~"[0-9]+"
    nl = ~"\\n"
    ws = ~"\\s"  
    """)

# Txt tree visitor
class TxtVisitor(NodeVisitor):
    def visit_txt(self, node, visited_children):
        j = visited_children[1]
        m = visited_children[3]
        t = visited_children[5]
        output = {}
        output.update({"jobs": j})
        output.update({"machines": m})
        output.update({"tasks": t})
        return output

    def visit_tasks(self, node, visited_children):
        t = node.text.strip()
        t = t.split("\n")
        t = [[int(i) for i in r.split()] for r in t]
        t = [[r[i:i+2] for i in range(0, len(r), 2)] for r in t]
        return t
    
    def visit_int(self, node, visited_children):
        return int(node.text)
    
    def generic_visit(self, node, visited_children):
        return None
    
# Parse txt
txt_filepath = sys.argv[1]
txt_file = open(txt_filepath, "r")
txt_tree = txt_grammar.parse(txt_file.read())
txt_file.close()

# Initialize json content
json_content = TxtVisitor().visit(txt_tree)
json_content_str = json.dumps(json_content, sort_keys=False, indent=4)

# Fix array indentation
json_content_str = re.sub("\s{8,}", "", json_content_str)
json_content_str = re.sub("\[\[\[", "[\n        [[", json_content_str)
json_content_str = re.sub("\]\],\[\[", "]],\n        [[", json_content_str)

# Write json
json_filepath = os.path.splitext(txt_filepath)[0] + ".json"
json_file = open(json_filepath, "w")
json_file.write(json_content_str)
json_file.close()
