import subprocess
import benchmarksCommon
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor 
  
# Output grammar
output_grammar = Grammar("""
    output = output_line*
    output_line = solution / info
    solution = "[SOLUTION] Time: " float " | Cost: " int " | Solution: [" list_int "]" nl
    info = "[INFO] " ~"."+ nl
    list_int = ~"[0-9,\\s]*"
    float = int "." int
    int = ~"[0-9]+"
    nl = ~"\\n"
    ws = ~"\\s"
    """)

# Output tree visitor
class OutputVisitor(NodeVisitor):
    def visit_output(self, node, visited_children):
        solutions = [solution for solution in visited_children if solution]
        if solutions:
            return solutions[-1]
        else:
            return None, None, None

    def visit_output_line(self, node, visited_children):
        return visited_children[0]
    
    def visit_solution(self, node, visited_children):
        search_time = visited_children[1]
        cost = visited_children[3]
        solution = visited_children[5]
        return cost, search_time, solution

    def visit_list_int(self, node, visited_children):
        l = node.text.split(",")
        if len(l) > 1:
            l = [int(i) for i in l]
        else:
            l = []
        return l
    
    def visit_float(self, node, visited_children):
        return float(node.text)
        
    def visit_int(self, node, visited_children):
        return int(node.text)
    
    def generic_visit(self, node, visited_children):
        return None
    
def solve(args, json_file):
    solver_config = benchmarksCommon.solvers_configs["mdd"]
    
    cmd = \
        [solver_config["path"]] + \
        ["-s"] + \
        ["-q"] + [str(args.q)] + \
        ["-t"] + [str(args.t)] + \
        ["--wc"] + [str(args.wc)] + \
        ["--pc"] + [str(args.pc)] + \
        ["--wg"] + [str(args.wg)] + \
        ["--pg"] + [str(args.pg)] + \
        ["--eq"] + [str(args.eq)] + \
        ["--neq"] + [str(args.neq)] + \
        [json_file]

    cost, search_time, solution = None, None, None
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="ascii")
    except subprocess.CalledProcessError:
        pass
    else:
        output = output.replace("\33[2K\n","") # Remove clear line
        output_tree = output_grammar.parse(output)
        cost, search_time, solution = OutputVisitor().visit(output_tree)
        
    return cost, search_time, solution

