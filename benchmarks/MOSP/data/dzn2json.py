import os
import re
import sys
import json
import jsbeautifier
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# Dzn grammar
dzn_grammar = Grammar("""
    dzn = clients ws* products ws* orders ws*
    clients = "c = " int ";"
    products = "p = " int ";"
    orders = "orders = [|" ws* matrix_int ws* "|]" ";"?
    matrix_int = list_int (ws* "|" ws* list_int)*  
    list_int = int (ws* "," ws* int)*
    int = ~"[0-9]+"
    ws = ~"\\s"   
    """)

# Dzn tree visitor
class DznVisitor(NodeVisitor):
    def visit_dzn(self, node, visited_children):
        output = {}
        output.update(visited_children[0])
        output.update(visited_children[2])
        output.update(visited_children[4])
        return output

    def visit_clients(self, node, visited_children):
        return {"clients": visited_children[1]}
    
    def visit_products(self, node, visited_children):
        return {"products": visited_children[1]}
    
    def visit_orders(self, node, visited_children):
        return {"orders": visited_children[2]}
        
    def visit_matrix_int(self, node, visited_children):
        m = re.sub("[|,]", " ", node.text)
        m = re.sub("\s+", " ", m)
        return [int(i) for i in m.split(" ")]
    
    def visit_int(self, node, visited_children):
        return int(node.text)
    
    def generic_visit(self, node, visited_children):
        return {}
    
# Parse dzn
dzn_filepath = sys.argv[1]
dzn_file = open(dzn_filepath, "r")
dzn_tree = dzn_grammar.parse(dzn_file.read())
dzn_file.close()

# Write json
json_content = DznVisitor().visit(dzn_tree)
json_content_str = json.dumps(json_content,sort_keys=False)
json_content_str = jsbeautifier.beautify(json_content_str)
json_filepath = os.path.splitext(dzn_filepath)[0] + ".json"
json_file = open(json_filepath, "w")
json_file.write(json_content_str)
json_file.close()
