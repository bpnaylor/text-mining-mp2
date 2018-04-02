import os
import re
import numpy as np
from sklearn.model_selection import KFold
from sys import argv
import time
import pandas as pd
from matplotlib import pyplot as plt
import random

def main():
	start_time = time.time()

	all_files = []
	file_path = argv[1]
	data_files = load_directory(os.path.abspath(file_path))
	# tune_parameters(data_files)
	cross_validate(data_files)

	print("Time: " + str(time.time() - start_time))

def load_directory(path):
	f = []
	for root, dirs, files in os.walk(path, topdown=False):
		for name in files:
			f.append(open(os.path.join(root, name)))
		for name in dirs:
			loadDirectory(os.path.join(root, name))
	return(f)

def tokenize(file, doc_id):
	s = file.read()
	s = s.replace("="," ").replace("["," ").replace("]"," ")
	s = re.sub("(?<![\r\n])(\r?\n|\n?\r)(?![\r\n])", " ", s)
	s = re.sub(r'([\n])\1+', r' \1', s)
	s = s.lower().replace("\n","START ").split(" ")
	s = list(filter(None, s))    # remove empty strings

	for i in range(len(s)):
		if s[i].count("/") > 1:
			s[i] = s[i].replace("\/"," ", 1)

	tokens = []
	for i in range(len(s)-1):
		if not (s[i] == "START" and s[i+1] == "START"):
			tokens.append(s[i])
	
	if tokens[len(tokens)-1] == "START":
		tokens = tokens[:-1]

	file.seek(0)

	return(tokens)

def build_model(tokens):
	prev_tag = tokens[0]

	for i in range(1,len(tokens)):
		curr_tag = tokens[i]     # ... for now ;)
		
		if tokens[i] != "START":
			curr_token = tokens[i].split("/")
			curr_word = curr_token[0]
			curr_tag = curr_token[1]

			check_dict(emissions, curr_word, curr_tag)
			check_dict(transitions, curr_tag, prev_tag)

		prev_tag = curr_tag

def tag_tokens(tokens, doc_id):
	untagged = []
	true_tags = []
	pred_tags = []
	num_sentences = 0

	for i in range(len(tokens)):
		if tokens[i] == "START":
			untagged.append([])
			true_tags.append([])
			num_sentences += 1
		else:
			token = tokens[i].split("/")
			untagged[num_sentences-1].append(token[0])		# word
			true_tags[num_sentences-1].append(token[1])		# word's true tag

	for sentence in range(num_sentences):
		tagged_sentence = viterbi(untagged[sentence], doc_id)

		pred_tags.append(tagged_sentence)

		for word in range(len(untagged[sentence])):
			true_tag = true_tags[sentence][word]
			pred_tag = pred_tags[sentence][word]
			curr_word = untagged[sentence][word]

			check_dict(confusion_dict, true_tag, pred_tag)

def check_dict(dictionary, a, b):
	if a in dictionary:
		if b in dictionary[a]:
			dictionary[a][b] = dictionary[a][b] + 1
		else:
			dictionary[a][b] = 1
	else:
		dictionary[a] = {}
		dictionary[a][b] = 1


def smooth_prob(dictionary, parameter, a, b):
	c = 1.0
	for d in dictionary[a]:
		c -= dictionary[a][d]

	dictionary[a][b] = abs(parameter*c)

def normalize_probabilities():
	global sigma
	global delta

	for word in emissions:
		c = 0
		for v in emissions[word].values():
			c += v + sigma
		for k in emissions[word].keys():
			emissions[word][k] /= c

	for tag in transitions:
		c = 0
		for v in transitions[tag].values():
			c += v + delta
		for k in transitions[tag].keys():
			transitions[tag][k] /= c

def viterbi(tokens, doc_id):
	global sigma
	tags = {}

	# find all possible tags for all tokens in sentence
	for i in range(len(tokens)):
		if tokens[i] in emissions:
			for k in emissions[tokens[i]]:
				tags[k] = 0
		else:
		# naive baseline: treat all unrecognized tokens as NNP or NNPS
			emissions[tokens[i]] = {}
			smooth_prob(emissions, sigma, tokens[i], "nnp")
			tags["nnp"] = 0

	tags = list(tags.keys())

	# nested
	return(forward(tokens, tags, doc_id))

def forward(tokens, tags, doc_id):
	global delta
	# build trellis of probabilities and indices of possible tags

	trellis = np.zeros((len(tokens), len(tags), 2), dtype=np.float64)

	# first token
	curr_word = tokens[0]
	for j in range(len(tags)):
		curr_tag = tags[j]

		if curr_tag in emissions[curr_word]:				
			prev_tag = "START"

			if curr_tag in transitions:
				if prev_tag not in transitions[curr_tag]:
					smooth_prob(transitions, delta, curr_tag, prev_tag)

				trellis[0,j,0] = emissions[curr_word][curr_tag] * transitions[curr_tag][prev_tag]
				trellis[0,j,1] = j

	# remaining tokens
	for i in range(1,len(tokens)):
		for j in range(0,len(tags)):
			curr_word = tokens[i]
			curr_tag = tags[j]

			if curr_word in emissions:
				if curr_tag in emissions[curr_word]:
					b = np.zeros(len(tags))

					for k in range(len(tags)):
						prev_tag = tags[k]
						curr_tag = tags[j]

						if curr_tag in transitions:
							if prev_tag not in transitions[curr_tag]:
								smooth_prob(transitions, delta, curr_tag, prev_tag)
							b[k] = trellis[i-1,k,0] * transitions[curr_tag][prev_tag] * emissions[curr_word][curr_tag]

					trellis[i,j,0] = max(b)
					trellis[i,j,1] = np.argmax(b)

	return(backward(tokens, tags, trellis))

def backward(tokens, tags, trellis):
	# find most likely tag sequence

	tag_seq = ["" for i in range(len(tokens))]
	
	# last token
	argmax = -1
	maximum = 0

	for k in range(len(tags)):
		if trellis[len(tokens)-1,k,0] > maximum:
			maximum = trellis[len(tokens)-1,k,0]
			argmax = k

	prev_index = int(argmax)
	tag_seq[len(tokens)-1] = tags[prev_index]

	# remaining tokens
	for i in range(len(tokens)-2, -1, -1):
		curr_index = int(trellis[i+1,prev_index,1])

		tag_seq[i] = tags[curr_index]
		prev_index = curr_index

	return(tag_seq)

def tune_parameters(files):
	global delta
	global sigma

	random.seed(0)

	parameters = np.linspace(0.01, 10, num=50, endpoint=True, retstep=True)
	parameters = parameters[0]

	print("Step size: " + str(parameters[1]))

	overalls = np.empty((len(parameters),len(parameters),3))

	for i in range(len(parameters)):
		for j in range(len(parameters)):
			delta = parameters[0]
			sigma = parameters[j]
			results = cross_validate(files)

			overalls[i,j,0] = results[0]	# accuracy
			overalls[i,j,1] = results[1]	# precision
			overalls[i,j,2] = results[2]	# recall

			print(str(delta) + ", " + str(sigma) + ": " + str(results[0] + results[1] + results[2]))

	# find joint argmax of f = accuracy + precision + recall
	delta, sigma = optimize(overalls, parameters)
	print("")
	print("Best delta = " + str(delta))
	print("Best sigma = " + str(sigma))
	cross_validate(files)

def optimize(overalls, parameters):		
	summation = 0.0
	maximize = 0.0

	for i in range(len(parameters)):
		for j in range(len(parameters)):
			summation = np.sum(overalls[i,j,:])
			if summation > maximize:
				delta_ind = i
				sigma_ind = j
				maximize = summation

	return(parameters[delta_ind], parameters[sigma_ind])


def cross_validate(files):
	kf = KFold(n_splits=5, shuffle=True, random_state=0)

	avg_accuracy = []
	avg_precision = []
	avg_recall = []

	count = 0

	for train_indices, test_indices in kf.split(files):
		for i in train_indices:
			tokens = tokenize(files[i],i)
			build_model(tokens)

		normalize_probabilities()

		for i in test_indices:
			tokens = tokenize(files[i],i)
			tag_tokens(tokens, i)

		# print_metrics(count)

		results = store_metrics()
		avg_accuracy.append(results[0])
		avg_precision.append(results[1])
		avg_recall.append(results[2])

		emissions.clear()
		transitions.clear()
		confusion_dict.clear()

		count += 1

	# print("Accuracy : " + str(np.mean(avg_accuracy)))
	# print("Precision : " + str(np.mean(avg_precision)))
	# print("Recall : " + str(np.mean(avg_recall)))

	return(np.mean(avg_accuracy), np.mean(avg_precision), np.mean(avg_recall))

def store_metrics():
	confusion_matrix = pd.DataFrame(confusion_dict).T.fillna(0)

	precision_d = np.sum(confusion_matrix,axis=0)
	recall_d = np.sum(confusion_matrix,axis=1)
	accuracy_d = np.sum(np.sum(confusion_matrix,axis=0))

	p = 0
	r = 0
	a = np.trace(confusion_matrix)/accuracy_d

	for t in confusion_matrix:
		if t in confusion_matrix[t]:
			p += confusion_matrix[t][t]/precision_d[t]
			r += confusion_matrix[t][t]/recall_d[t]

	p /= len(precision_d)
	r /= len(recall_d)

	return(a, p, r)

def print_metrics(count):
	confusion_matrix = pd.DataFrame(confusion_dict).T.fillna(0)

	precision_d = np.sum(confusion_matrix,axis=0)
	recall_d = np.sum(confusion_matrix,axis=1)
	accuracy_d = np.sum(np.sum(confusion_matrix,axis=0))

	tags = list(confusion_matrix.keys())

	for i in range(len(tags)):
		if tags[i] in confusion_matrix[tags[i]]:
			specifics[i,count,0] = confusion_matrix[tags[i]][tags[i]]/precision_d[tags[i]]
			specifics[i,count,1] = confusion_matrix[tags[i]][tags[i]]/recall_d[tags[i]]

	if count == 4:
		for i in range(len(tags)):
			print(tags[i])
			print("\t precision: " + str(np.mean(specifics[i,:,0])))
			print("\t recall: " + str(np.mean(specifics[i,:,1])))

# global variables for laziness
emissions = {}       # emissions["word"]["tag"] = p("word" | "tag")
transitions = {}     # transitions["tag_i"]["tag_j"] = p("tag_i" | "tag_j")
confusion_dict = {}

specifics = np.zeros((100,5,2))

global sigma
global delta

sigma = 1
delta = 0.01

main()