import os
import sys
import json
import jsbeautifier
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# Dzn grammar
dzn_grammar = Grammar("""
    dzn = clients ws products ws orders ws
    clients = "c" ws "=" ws int ";"
    products = "p" ws "=" ws int ";"
    orders = "orders" ws "=" ws "[" ws "|" ws matrix_int ws "|" ws "]" ";"?
    matrix_int = list_int (ws "|" ws list_int)*  
    list_int = int (ws "," ws int)*
    int = ~r"[0-9]+"
    ws = ~r"\\s*"  
    """)

# Dzn tree visitor
class DznVisitor(NodeVisitor):
    def visit_dzn(self, node, visited_children):
        output = {}
        for child in visited_children:
            output.update(child)
        return output

    def visit_clients(self, node, visited_children):
        c = int(visited_children[4])
        return {"clients": c}
    
    def visit_products(self, node, visited_children):
        p = visited_children[4]
        return {"products": p}
    
    def visit_orders(self, node, visited_children):
        o = visited_children[8]
        return {"orders": o}
        
    def visit_matrix_int(self, node, visited_children):
        rows_text = node.text.split("|")
        rows = []
        for row_text in rows_text:
            rows.append([int(i) for i in row_text.split(",")])
        return rows
    
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
