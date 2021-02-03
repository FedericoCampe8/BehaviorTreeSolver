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
 
    file.write("{\n")
    file.write("\t\"k\": " + str(k) + ",\n")
    file.write("\t\"b\": " + str(b) + ",\n")

    atomicsFirst = ",\n\t\t".join([(x[0] + "," + x[1]) for x in atomics])    
    if len(atomics) > 0:
        file.write("\t\"AtomicConstraints\": [\n\t\t" + atomicsFirst + "\n\t],\n");
    else:
        file.write("\t\"AtomicConstraints\": [],\n");
    
    disA1 = ",\n\t\t".join([(x[0] + "," + x[1] + "," + x[2] + "," + x[3]) for x in disjuncts])    
    if len(disjuncts) > 0:
        file.write("\t\"DisjunctiveConstraints\": [\n\t\t" + disA1 + "\n\t],\n");
    else:
        file.write("\t\DisjunctiveConstraints\": [],\n");
    
    satomicsFirst = ",\n\t\t".join([(x[0] + "," + x[1]) for x in softAtomics])
    if len(softAtomics) > 0:
        file.write("\t\"SoftAtomicConstraints\": [\n\t\t" + satomicsFirst + "\n\t],\n");
    else:
        file.write("\t\"SoftAtomicConstraints\": [],\n");    

    ds = ",\n\t\t".join([x for x in directSucc])    
    if len(ds) > 0:
        file.write("\t\"DirectSuccessors\": [\n\t\t" + ds + "\n\t],\n");
    else:
        file.write("\t\"DirectSuccessors\": []\n");
    file.write("}")
    file.close()
    
    
# Main
import sys

atomics, disjuncts, softAtomics, directSucc, k, b = read_file(sys.argv[1])
write_file(sys.argv[2], atomics, disjuncts, softAtomics, directSucc, k, b)
