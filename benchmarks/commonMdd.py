import time
import argparse
import subprocess
from common import Result, Solvers
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",    type=int, action="store", dest="t",   default=60)
    parser.add_argument("-q",    type=int, action="store", dest="q",   default=50000)
    parser.add_argument("--wc",  type=int, action="store", dest="wc",  required=True)
    parser.add_argument("--pc",  type=int, action="store", dest="pc",  required=True)
    parser.add_argument("--wg",  type=int, action="store", dest="wg",  required=True)
    parser.add_argument("--pg",  type=int, action="store", dest="pg",  required=True)
    parser.add_argument("--eq",  type=float, action="store", dest="eq",  default=0)
    parser.add_argument("--neq", type=float, action="store", dest="neq", default=25)
    args, _ = parser.parse_known_args(argv)
    return args


def get_args_str(args):
    flat_args = args.t, args.q, args.wc, args.pc, args.wg, args.pg, args.eq, args.neq
    return "t{}-q{}-wc{}-pc{}-wg{}-pg{}-eq{}-neq{}-{}".format(*flat_args, int(time.time()))


def solve(args, json_file):
    cmd = \
        [Solvers["mdd"]["path"]] + \
        ["-s"] + \
        ["-q"] + [str(args.q)] + \
        ["-t"] + [str(args.t)] + \
        ["--wc"] + [str(args.wc)] + \
        ["--pc"] + [str(args.pc)] + \
        ["--wg"] + [str(args.wg)] + \
        ["--pg"] + [str(args.pg)] + \
        ["--eq"] + [str(args.eq)] + \
        ["--neq"] + [str(args.neq)] + \
        ["--rs"] + ["0"] + \
        [json_file]

    results = []
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding="ascii")
    except subprocess.CalledProcessError:
        pass
    else:
        output = output.replace("\33[2K\n","") # Remove clear line
        output_tree = output_grammar.parse(output)
        results = OutputVisitor().visit(output_tree)
    return results


### Output parsing
output_grammar = Grammar("""
    output = output_line+
    output_line = solution / info
    solution = "[SOLUTION] Source: " source " | Time: " time " | Iteration: " int " | Cost: " int " | Solution: " list_int nl
    source = "CPU" / "GPU"
    info = "[INFO]" ~"."+ nl
    list_int = "[" ~"[0-9,\\s]*" "]"
    time = int ":" int ":" int "." int
    int = ~"[0-9]+"
    nl = ~"\\n"
    ws = ~"\\s"
    """)

class OutputVisitor(NodeVisitor):
    def visit_output(self, node, visited_children):
        return [solution for solution in visited_children if solution]

    def visit_output_line(self, node, visited_children):
        return visited_children[0]

    def visit_solution(self, node, visited_children):
        cost = visited_children[7]
        solution = visited_children[9]
        return Result(cost, solution)

    def visit_list_int(self, node, visited_children):
        l = node.children[1].text.split(",")
        if len(l) > 1:
            l = [int(i) for i in l]
        else:
            l = []
        return l

    def visit_int(self, node, visited_children):
        return int(node.text)

    def generic_visit(self, node, visited_children):
        return None