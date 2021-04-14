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
  
# Timeout
timeout = 45

# Instances
instances = [
	"ft06",
	"la04",
	"la03",
	"la05",
	"la02",
	"la01",
	"la08",
	"la10",
	"la06",
	"la09",
	"la07",
	"orb08",
	"la12",
	"la17",
	"la14",
	"orb09",
	"orb01",
	"la19",
	"la15",
	"la20",
	"ft20",
	"ft10",
	"orb10",
	"orb07",
	"orb06",
	"la11",
	"orb04",
	"la16",
	"la13",
	"orb05",
	"orb02",
	"la18",
	"orb03",
	"abz6",
	"abz5",
	"la22",
	"la25",
	"la24",
	"la23",
	"la21",
	"swv05",
	"swv01",
	"swv03",
	"la28",
	"swv04",
	"la29",
	"swv02",
	"la30",
	"la27",
	"la26",
	"ta08",
	"ta10",
	"ta06",
	"ta04",
	"ta01",
	"la38",
	"ta05",
	"ta09",
	"ta02",
	"ta07",
	"ta03",
	"la40",
	"la39",
	"la37",
	"la36",
	"la33",
	"la35",
	"la32",
	"la34",
	"la31",
	"ta13",
	"ta15",
	"ta18",
	"ta17",
	"swv08",
	"swv06",
	"swv07",
	"ta16",
	"ta20",
	"ta12",
	"ta19",
	"ta11",
	"swv09",
	"ta14",
	"swv10",
	"abz9",
	"abz8",
	"abz7",
	"ta25",
	"ta21",
	"ta28",
	"ta30",
	"ta24",
	"ta22",
	"ta23",
	"ta29",
	"ta27",
	"ta26",
	"yn4",
	"yn3",
	"yn2",
	"yn1",
	"ta31",
	"ta35",
	"ta40",
	"ta38",
	"ta34",
	"ta39",
	"ta36",
	"ta32",
	"ta37",
	"ta33",
	"swv13",
	"swv17",
	"swv20",
	"swv14",
	"swv12",
	"swv16",
	"swv18",
	"swv11",
	"swv19",
	"swv15",
	"ta43",
	"ta48",
	"ta45",
	"ta49",
	"ta46",
	"ta41",
	"ta44",
	"ta42",
	"ta50",
	"ta47",
	"ta52",
	"ta59",
	"ta53",
	"ta56",
	"ta57",
	"ta60",
	"ta54",
	"ta55",
	"ta51",
	"ta58",
	"ta63",
	"ta64",
	"ta70",
	"ta61",
	"ta68",
	"ta67",
	"ta66",
	"ta65",
	"ta62",
	"ta69",
	"ta78",
	"ta80",
	"ta76",
	"ta73",
	"ta77",
	"ta72",
	"ta79",
	"ta71",
	"ta75",
	"ta74"]
