import os
import re
import numpy as np
# import sklearn.metrics as sk
from sklearn.model_selection import KFold
from sys import argv
import time
import pandas as pd
from matplotlib import pyplot as plt
import random
from operator import itemgetter
from math import log

def sample_next_tag(prev_tag, p):
	options = list(transitions[prev_tag].keys())
	probs = list(t+0.001 for t in list(transitions[prev_tag].values()))
	return(random.choices(population=options, weights=probs, k=1)[0])

def sample_next_word(curr_tag, p):
	options = list(emissions[curr_tag].keys())
	probs = list(e+0.005 for e in list(emissions[curr_tag].values()))
	return(random.choices(population=options, weights=probs, k=1)[0])

def sample_sentence(length):
	prev_tag = "START"
	words = []
	tags = []

	for i in range(length):
		curr_tag = sample_next_tag(prev_tag, -1.0)

		words.append(sample_next_word(curr_tag, -1.0))
		tags.append(curr_tag)

		prev_tag = curr_tag

	return(words, tags)

# TODO: generate 100 sentences from trained HMM
	# record tag sequence, word sequence, log-likelihood for each sentence
	# more natural/readable than mp1?

def generate_sentences(num_sentences):
	random.seed(1)

	word_sequences = []
	tag_sequences = []
	log_likelihoods = []
	sentence_length = 10

	for s in range(num_sentences):
		sentence = sample_sentence(sentence_length)
		word_sequences.append(sentence[0])
		tag_sequences.append(sentence[1])
		log_likelihoods.append([calculate_loglikelihood(word_sequences[s],tag_sequences[s]),s])

	# sorted_ll = sorted(log_likelihoods, key=itemgetter(0), reverse=True)

	# for i in range(100):
	# 	j = sorted_ll[i][1]

		# print(" ".join(word_sequences[j]))
		# print(" ".join(tag_sequences[j]))
		# print(sorted_ll[i][0])
		# print("")

	tag_generated_sentences(word_sequences, tag_sequences, log_likelihoods)

def tag_generated_sentences(words, tags, lls):
	print("accuracy,log-likelihood")
	overall = 0
	for i in range(len(words)):
		sentence = words[i]
		true_tags = tags[i]
		log_likelihood = lls[i]
		accuracy = 0

		pred_tags = viterbi(sentence)

		for j in range(len(sentence)):
			if pred_tags[j] == true_tags[j]:
				accuracy+=1
				overall+=1

		print(str(accuracy/10.0) + "," + str(lls[i][0]))
		print(overall/1000.0)

def calculate_loglikelihood(words, tags):
	ll = 0
	prev = "START"

	for i in range(len(words)):
		w = words[i]
		curr = tags[i]

		if emissions[curr][w]*transitions[prev][curr] > 0:
			ll += log(emissions[curr][w]*transitions[prev][curr])

		prev = curr

	return(ll)

# TODO: run viterbi algorithm on generated sentences
	# evaluate inferred tags against generated tags
	# report average tagging accuracy over all sentences
	# correlation between sentence-level tagging accuracy and log-likelihood of each sentence?

def main():
	start_time = time.time()

	all_files = []
	file_path = argv[1]
	data_files = load_directory(os.path.abspath(file_path))
	tag_all(data_files)

	# cross_validate(data_files)
	generate_sentences(100)

	print("Time: " + str(time.time() - start_time))

def load_directory(path):
	f = []
	for root, dirs, files in os.walk(path, topdown=False):
		for name in files:
			f.append(open(os.path.join(root, name)))
		for name in dirs:
			loadDirectory(os.path.join(root, name))
	return(f)

def tag_all(files):
	for i in range(len(files)):
		tokens = tokenize(files[i], i)	# gets list of strings which are “START” or “word/tag”
		build_model(tokens)

		normalize_probabilities()

def cross_validate(files):
	kf = KFold(n_splits=5, shuffle=True, random_state=0)

	for train_indices, test_indices in kf.split(files):
		for i in train_indices:
			tokens = tokenize(files[i],i)
			build_model(tokens)

		normalize_probabilities()

		for i in test_indices:
			tokens = tokenize(files[i],i)
			tag_tokens(tokens, i)

		emissions.clear()
		transitions.clear()

def tokenize(file, doc_id):
	s = file.read()
	s = s.replace("="," ").replace("["," ").replace("]"," ")
	s = re.sub("(?<![\r\n])(\r?\n|\n?\r)(?![\r\n])", " ", s)
	s = re.sub(r'([\n])\1+', r' \1', s)
	s = s.lower()
	s = s.replace("\n","START ")
	s = s.split(" ")
	s = list(filter(None, s))    # remove empty strings

	for i in range(len(s)):
		if s[i].count("/") > 1:
			s[i] = s[i].replace("/"," ", 1)

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

			check_dict(emissions, curr_tag, curr_word)
			check_dict(transitions, prev_tag, curr_tag)

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
		tagged_sentence = viterbi(untagged[sentence])

		pred_tags.append(tagged_sentence)

		for word in range(len(untagged[sentence])):
			true_tag = true_tags[sentence][word]
			pred_tag = pred_tags[sentence][word]
			curr_word = untagged[sentence][word]

			# if true_tag != pred_tag:

def check_dict(dictionary, a, b):
	if a in dictionary:
		if b in dictionary[a]:
			dictionary[a][b] = dictionary[a][b] + 1
		else:
			dictionary[a][b] = 1
	else:
		dictionary[a] = {}
		dictionary[a][b] = 1

def viterbi(tokens):
	tags = {}

	# find all possible tags for all tokens in sentence
	for i in range(len(tokens)):
		for k in emissions:
			if tokens[i] in emissions[k]:
				tags[k] = 0
			else:
				# naive baseline: treat all unseen tokens as NNP
				smooth_prob(emissions, "nnp", tokens[i])
				tags["nnp"] = 0

	tags = list(tags.keys())

	# nested
	return(forward(tokens, tags))

def forward(tokens, tags):
	# build trellis of probabilities and indices of possible tags

	trellis = np.zeros((len(tokens), len(tags), 2), dtype=np.float64)

	# first token
	curr_word = tokens[0]
	prev_tag = "START"
	for j in range(len(tags)):
		curr_tag = tags[j]

		if curr_word in emissions[curr_tag]:
			if curr_tag not in transitions[prev_tag]:
				smooth_prob(transitions, prev_tag, curr_tag)

			trellis[0,j,0] = emissions[curr_tag][curr_word] * transitions[prev_tag][curr_tag]
			trellis[0,j,1] = j

	# remaining tokens
	for i in range(1,len(tokens)):
		for j in range(0,len(tags)):
			curr_word = tokens[i]
			curr_tag = tags[j]

			if curr_tag in emissions:
				if curr_word in emissions[curr_tag]:
					b = np.zeros(len(tags))

					for k in range(len(tags)):
						prev_tag = tags[k]
						curr_tag = tags[j]

						if prev_tag in transitions:
							if curr_tag not in transitions[prev_tag]:
								smooth_prob(transitions, prev_tag, curr_tag)
							b[k] = trellis[i-1,k,0] * transitions[prev_tag][curr_tag] * emissions[curr_tag][curr_word]

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

def smooth_prob(dictionary, a, b):
	global delta
	c = 1.0
	for k in dictionary[a]:
		c -= dictionary[a][k]

	dictionary[a][b] = abs(delta*c)

def normalize_probabilities():
	global sigma
	global delta

	for tag in emissions:
		c = 0
		for v in emissions[tag].values():
			v += sigma
			c += v
		for k in emissions[tag].keys():
			emissions[tag][k] /= c

	for tag in transitions:
		c = 0
		for v in transitions[tag].values():
			v += delta
			c += v
		for k in transitions[tag].keys():
			transitions[tag][k] /= c

# global variables for laziness

emissions = {}       # emissions["tag"]["word"] = p("word" | "tag")
transitions = {}     # transitions["tag_i"]["tag_j"] = p("tag_j" | "tag_i")

global sigma
global delta
sigma = 0.1
delta = 0.5

main()
