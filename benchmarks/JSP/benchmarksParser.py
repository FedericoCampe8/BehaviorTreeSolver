import os
import re
import sys
import json
import numpy as np
import jsbeautifier
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# Output grammar
OutputGrammar = Grammar("""
    output = ws* solutions ws* "==========" ws*
    solutions = solution*
    solution = "Cost: " int " | Time: " float " | Solution: " int_list ws* "----------" ws*
    float = int "." int
    int = ~"[0-9]+"
    int_list = "[" ws* int ws* ("," ws int)* ws* "]"
    ws = ~"\\s"  
    """)

# Output tree visitor
class OutputVisitor(NodeVisitor):
    def visit_output(self, node, visited_children):
        if visited_children[1] != None:
            return visited_children[1][-1]
        else:
            return None, None, None
        
    def visit_solutions(self, node, visited_children): 
        return visited_children
    
    def visit_solution(self, node, visited_children):        
        cost = visited_children[1]
        time = visited_children[3]
        solution = visited_children[5]
        return cost, time, solution

    def visit_int(self, node, visited_children):
        return int(node.text)
    
    def visit_float(self, node, visited_children):
        return float(node.text)
    
    def visit_int_list(self, node, visited_children):
        l = re.sub("[\[\],]", " ", node.text)
        l = re.sub("\s+", " ", l)
        l = l.strip()
        return [int(i) for i in l.split()]
    
    def generic_visit(self, node, visited_children):
        return None
