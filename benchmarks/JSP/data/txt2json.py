import os
import re
import sys
import json
import numpy as np
import jsbeautifier
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
        t = [t[i:i+2] for i in range(0, len(t), 2)] 
        output = {}
        output.update({"jobs": j})
        output.update({"machines": m})
        output.update({"tasks": t})
        return output

    def visit_tasks(self, node, visited_children):
        t = re.sub("\s+", " ", node.text)
        return [int(i.strip()) for i in t.split()]
    
    def visit_int(self, node, visited_children):
        return int(node.text)
    
    def generic_visit(self, node, visited_children):
        return {}
    
# Parse txt
txt_filepath = sys.argv[1]
txt_file = open(txt_filepath, "r")
txt_tree = txt_grammar.parse(txt_file.read())
txt_file.close()

# Write json
json_content = TxtVisitor().visit(txt_tree)
json_content_str = json.dumps(json_content,sort_keys=False)
json_content_str = jsbeautifier.beautify(json_content_str)
json_filepath = os.path.splitext(txt_filepath)[0] + ".json"
json_file = open(json_filepath, "w")
json_file.write(json_content_str)
json_file.close()
