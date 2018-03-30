import os
from sys import argv

all_files = []
emissions = {}       # "word/tag"      -> c("word/tag")
transitions = {}     # "tag_i/tag_j"   -> c("tag_i/tag_j")

def load_directory(path):
    f = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            f.append(open(os.path.join(root, name)))
        for name in dirs:
            loadDirectory(os.path.join(root, name))
    return(f)

def tokenize(file):
    s = file.read()
    s = s.replace("[ ","").replace("=","").replace(" ]"," ").replace("\n\n\n","START ").replace("\n\n", "START ").replace("\n"," ").replace("  "," ").split(" ")
    s = list(filter(None, s))    # remove empty strings
    return(s)

def build_model(tokens, doc_id):
    prev_tag = tokens[0]
    
    for i in range(1,len(tokens)):
        
        curr_tag = tokens[i]     # ... for now ;)
        
        if len(tokens[i].split("/")) > 1:
            try:
                emissions[tokens[i]] = emissions[tokens[i]] + 1
            except KeyError as e:
                emissions[tokens[i]] = 1
                
            curr_tag = tokens[i].split("/")[1]
            try:
                transitions[prev_tag + "/" + curr_tag] = transitions[prev_tag + "/" + curr_tag] + 1 
            except KeyError as e:
                transitions[prev_tag + "/" + curr_tag] = 1

        prev_tag = curr_tag

def tokenize_all(files):
    for i in range(len(files)):
        tokens = tokenize(files[i])        
        build_model(tokens, i)

file_path = argv[1]
data_files = load_directory(os.path.abspath(file_path))
tokenize_all(data_files)

emissions_sorted = [(k, emissions[k]+0.1) for k in sorted(emissions, key=emissions.get, reverse=True)]
transitions_sorted = [(k, transitions[k]+0.5) for k in sorted(transitions, key=transitions.get, reverse=True)]

i=0
for k, v in emissions_sorted:
    if "/NN" in k and "/NNP" not in k and "/NNS" not in k:
        print(k, v)
        i+=1
    if i==10:
        break

print()

for k, v in transitions_sorted:
    if "VB/" in k:
        print(k, v)
        i+=1
    if i==20:
        break

