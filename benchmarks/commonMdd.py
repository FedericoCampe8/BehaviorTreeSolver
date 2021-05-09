import argparse
import subprocess
from common import Result, Solvers
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",    type=int, action="store", dest="t")
    parser.add_argument("-i",    type=int, action="store", dest="t")
    parser.add_argument("-q",    type=int, action="store", dest="q")
    parser.add_argument("--wc",  type=int, action="store", dest="wc")
    parser.add_argument("--pc",  type=int, action="store", dest="pc")
    parser.add_argument("--wg",  type=int, action="store", dest="wg")
    parser.add_argument("--pg",  type=int, action="store", dest="pg")
    parser.add_argument("--eq",  type=int, action="store", dest="eq")
    parser.add_argument("--neq", type=int, action="store", dest="neq")
    args, _ = parser.parse_known_args(argv)
    return args


def get_args_str(args):
    flat_args = args.t, args.i, args.q, args.wc, args.pc, args.wg, args.pg, args.eq, args.neq
    return "t{}-i{}-q{}-wc{}-pc{}-wg{}-pg{}-eq{}-neq{}-{}".format(prefix, *flat_args, int(time.time()))


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
    output = output_line*
    output_line = solution / info
    solution = "[SOLUTION] Time: " float " | Cost: " int " | Solution: " list_int nl
    info = "[INFO]" ~"."+ nl
    list_int = "[" ~"[0-9,\\s]*" "]"
    float = int "." int
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
        search_time = visited_children[1]
        cost = visited_children[3]
        solution = visited_children[5]
        return Result(cost, search_time, solution)

    def visit_list_int(self, node, visited_children):
        l = node.children[1].text.split(",")
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
