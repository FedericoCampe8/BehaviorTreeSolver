# converts dat files to dzn files

import copy
import os
from typing import List

def read_file(filepath: str):
    atomics = []
    disjuncts = []
    softAtomics = []
    directSucc = []
    k = -1
    b = -1
    
    with open(filepath) as f:

        readAtomics = False
        readDisjuncts = False
        readDirectSucc = False
        readSoftAtomic = False

        for line in f:
            line = line.replace("<","[")
            line = line.replace(">,","]")
            line = line.replace(" ","")
            line = line.strip()        

            if "};" in line:
                readAtomics = False
                readDisjuncts = False
                readDirectSucc = False
                readSoftAtomic = False

            line = line.replace(";","")
            line = line.strip()

            if line.startswith("k"):
                line = line.replace("k=", "")
                k = int(line)
                
            if line.startswith("b"):
                line = line.replace("b=","")
                b = int(line)
                
            if line.startswith("chamber_sets_size"):
                line = line.replace("chamber_sets_size=","")

            if readAtomics:                
                atom = line.split(",")
                atomics.append((atom[0],atom[1]))

            if readDisjuncts:
                dis = line.split(",")
                disjuncts.append((dis[0],dis[1],dis[2],dis[3]))

            if readSoftAtomic:
                atom = line.split(",")
                softAtomics.append((atom[0],atom[1]))

            if readDirectSucc:
                atom = line.split(",")
                directSucc.append(atom[0])

            if line.startswith("AtomicConstr"):
                readAtomics = True
            if "Disjunctive" in line:
                readDisjuncts = True
            if "SoftAtomic" in line:
                readSoftAtomic = True
            if "DirectSuccessors" in line:
                readDirectSucc = True

        return atomics, disjuncts, softAtomics, directSucc, k, b


def write_file(filepath: str,
               atomics: List,
               disjuncts: List,
               softAtomics: List,
               directSucc: List,
               k: int,
               b: int):
    file = open(filepath, "w")
 
    file.write("digraph G\n")
    file.write("{\n")
    
    file.write("\n\t")
    for i in range(1,k+1):
        file.write(str(i) + ";")
    file.write("\n\n")
    
    file.write("\t{\n")        
    file.write("\t\tedge [color=green;weight=100000];\n")
    for x in atomics:
        file.write("\t\t" + x[0][1:] + " -> " + x[1][:-1] + ";\n")
    file.write("\t}\n\n")  
    
    file.write("\t{\n")        
    file.write("\t\tedge [weight=0];\n")
    color = 1
    for x in disjuncts:
        if x[0][1:] == x[3][:-1]:
            color = 1
        else:
            color = 2
        file.write("\t\t" + x[0][1:] + " -> " + x[1] + "[colorscheme=\"dark28\";color=" + str(1 + color) + ";];\n")
        file.write("\t\t" + x[2] + " -> " + x[3][:-1] + "[colorscheme=\"dark28\";color=" + str(1 + color) +  ";];\n")
    file.write("\t}\n")  
    
    file.write("}")
    file.close()
    
    
# Main
import sys

atomics, disjuncts, softAtomics, directSucc, k, b = read_file(sys.argv[1])
write_file(sys.argv[2], atomics, disjuncts, softAtomics, directSucc, k, b)
