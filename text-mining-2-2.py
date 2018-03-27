import os
import re
import numpy as np
import sklearn.metrics as sk

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
    s = s.replace("[ ","").replace("=","").replace(" ]"," ").replace("\n\n\n","START ").replace("\n\n", "START ").replace("\n"," ").replace("  "," ").replace("(","").replace(")","").replace("*","").split(" ")
    s = list(filter(None, s))    # remove empty strings
    return(s)

def build_model(tokens):
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
        build_model(tokens)
        tag_tokens(tokens, i)

def tag_tokens(tokens, doc_id):
    untagged = []
    true_tags = []
    est_tags = []
    num_sentences = 0

    for i in range(len(tokens)):
        if tokens[i] == 'START':
            untagged.append([])
            true_tags.append([])
            num_sentences += 1

        else:
            untagged[num_sentences-1].append(tokens[i].split("/")[0])
            true_tags[num_sentences-1].append(tokens[i].split("/")[1])
            n[0]+=1

    for i in range(num_sentences):
        est_tags.append(viterbi(untagged[i]))

        for j in range(len(est_tags[i])):
            if true_tags[i][j] == est_tags[i][j]:
                n[1]+=1
            else:
                print(untagged[i][j] + " -- " + true_tags[i][j] + " vs " + est_tags[i][j])
                if untagged[i][j] == "Philippine":
                    print(doc_id)

def viterbi(tokens):
    tags = {}
    tag_seq = ["" for i in range(len(tokens))]
    
    # find all possible tags for all tokens in sentence
    for i in range(len(tokens)):
        for k in [key.split("/")[1] for key, value in emissions.items() if re.match(r'^' + tokens[i] + r'/', key)]:
            tags[k] = 1
    tags = list(tags.keys())

    # forward: build trellis with index tracking
    trellis = np.zeros((len(tokens), len(tags), 2))

    # first token
    for j in range(len(tags)):
        e = tokens[0] + "/" + tags[j]
        if e in emissions:
            t = "START" + "/" + tags[j]
            if t in transitions:
                trellis[0,j,0] = emissions[e]*transitions[t]
                trellis[0,j,1] = j

    for i in range(1,len(tokens)):
        for j in range(0,len(tags)):
            e = tokens[i] + "/" + tags[j]

            if e in emissions:
                b = np.zeros(len(tags))

                for k in range(len(tags)):
                    t = tags[k] + "/" + tags[j]

                    if t in transitions:
                        b[k] = trellis[i-1,k,0] * transitions[t] * emissions[e]

                trellis[i,j,0] = max(b)
                trellis[i,j,1] = np.argmax(b)
    
    # backward: find most likely tag sequence
    prev_index = int(np.argmax([trellis[len(tokens)-1,k,0] for k in range(len(tags))]))
    tag_seq[len(tokens)-1] = tags[prev_index]

    for i in range(len(tokens)-2, -1, -1):
        curr_index = int(trellis[i+1,prev_index,1])

        tag_seq[i] = tags[curr_index]
        prev_index = curr_index

    return(tag_seq)

all_files = []
emissions = {}       # "word/tag"      -> c("word/tag")
transitions = {}     # "tag_i/tag_j"   -> c("tag_i/tag_j")
n = []
n.append(0)          # num tokens
n.append(0)          # accuracy

data_files = load_directory("C:/Users/redwa_000/Downloads/Max/0 UVa/Text Mining/text-mining-2/tagged")
tokenize_all(data_files)

print(n[1]/n[0])