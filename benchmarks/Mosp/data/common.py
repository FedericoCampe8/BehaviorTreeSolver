from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# Txt grammar
txt_grammar = Grammar("""
    txt = (description nl)* int ws+ int ws+ orders
    description =  ~"\D" ~"."+
    orders = (ws* int ws*)*  
    int = ~"[0-9]+"
    nl = ~"\\n"
    ws = ~"\\s"  
    """)

# Txt tree visitor
class TxtVisitor(NodeVisitor):
    def visit_txt(self, node, visited_children):
        c = visited_children[1]
        p = visited_children[3]
        o = visited_children[5]
        output = {}
        output.update({"clients": c})
        output.update({"products": p})
        output.update({"orders": o})
        return output

    def visit_orders(self, node, visited_children):
        t = node.text.strip()
        t = t.split("\n")
        t = [[int(i) for i in r.split()] for r in t]
        return t
    
    def visit_int(self, node, visited_children):
        return int(node.text)
    
    def generic_visit(self, node, visited_children):
        return None
    
