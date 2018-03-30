import os
import re
import numpy as np
# import sklearn.metrics as sk
from sklearn.model_selection import KFold
from sys import argv
import time

###### Part 1 - Parameter Estimation ###### 

# TODO: clean up regex parsing
	# words being labeled as tags
	# formatting change partway through files - parse differently based on file index?

###### Part 2 - Tag Sequence Inference ######

# TODO: build confusion matrix instead of precision/recall/accuracy dictionaries
	# implement confusion matrix
	# implement precision / recall / accuracy calculation from confusion matrix
		# Resources:
			# http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
			# https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
			# https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

	# report overall accuracy/precision/recall over all tags
	# report precision/recall for "NN", "VB", "JJ", "NNP"

# TODO: tune smoothing parameters sigma, delta; range = [0.1, 10]

###### Part 3 - Generate Sentence via HMM ######

# TODO: sample curr tag based on prev tag from transition probabilities
	# start with START dummy tag; sentence length = 10

# TODO: sample curr word from curr tag based on curr tag from emission probabilities

# TODO: generate 100 sentences from trained HMM
	# record tag sequence, word sequence, log-likelihood for each sentence
	# more natural/readable than mp1?

# TODO: run viterbi algorithm on generated sentences
	# evaluate inferred tags against generated tags
	# report average tagging accuracy over all sentences
	# correlation between sentence-level tagging accuracy and log-likelihood of each sentence?

def main():
	start_time = time.time()

	all_files = []
	file_path = argv[1]
	data_files = load_directory(os.path.abspath(file_path))
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

def cross_validate(files):
	sigma = 0.1
	delta = 0.5

	kf = KFold(n_splits=5, shuffle=True, random_state=0)

	for train_indices, test_indices in kf.split(files):
		for i in train_indices:
			tokens = tokenize(files[i])       
			build_model(tokens)

		normalize_probabilities(sigma, delta)

		for i in test_indices:
			tokens = tokenize(files[i])
			tag_tokens(tokens)

		# for true in recall:
		# 	d = 0
		# 	for est in recall[true]:
		# 		d += recall[true][est]
		# 	if true in recall[true]:
		# 		print("Recall of " + true + ": " + str(recall[true][true]/d))
		# 	else:
		# 		print("Recall of " + true + ": " + str(0.0))

		# for est in precision:
		# 	d = 0
		# 	for true in precision[est]:
		# 		d += precision[est][true]
		# 	if est in precision[est]:
		# 		print("Precision of " + est + ": " + str(precision[est][est]/d))
		# 	else:
		# 		print("Precision of " + est + ": " + str(0.0))

		emissions.clear()
		transitions.clear()

		# precision.clear()
		# recall.clear()

def tokenize(file):
	s = file.read()
	s = s.replace("[ ","").replace("=","").replace(" ]"," ").replace("\n\n\n\n\n","START ").replace("\n\n\n\n","START ").replace("\n\n\n","START ").replace("\n\n", "START ").replace("\n"," ").replace("  "," ").replace("(","").replace(")","").replace("*","").split(" ")
	s = list(filter(None, s))    # remove empty strings
	file.seek(0)
	return(s)

def build_model(tokens):
	prev_tag = tokens[0]

	for i in range(1,len(tokens)):
		curr_tag = tokens[i]     # ... for now ;)
		
		if len(tokens[i].split("/")) > 1:
			curr_token = tokens[i].split("/")
			curr_word = curr_token[0]
			curr_tag = curr_token[1]

			if curr_tag != ".START":
				check_dict(emissions, curr_word, curr_tag)

			if not (prev_tag == "START" and curr_tag == "START"):
				check_dict(transitions, prev_tag, curr_tag)

		prev_tag = curr_tag

def tag_tokens(tokens):
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
			token = tokens[i].split("/")
			untagged[num_sentences-1].append(token[0])		# word
			true_tags[num_sentences-1].append(token[1]) 	# word's true tag

	for sentence in range(num_sentences):
		tagged_sentence = viterbi(untagged[sentence])

		if tagged_sentence != -1:
			est_tags.append(tagged_sentence)

		for word in range(len(untagged[sentence])):
			true_tag = true_tags[sentence][word]
			est_tag = est_tags[sentence][word]

			# check_dict(recall, true_tag, est_tag)
			# check_dict(precision, est_tag, true_tag)

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
	if len(tokens) == 0:
		return(-1)

	tags = {}

	# find all possible tags for all tokens in sentence
	for i in range(len(tokens)):
		if tokens[i] in emissions:
			for k in emissions[tokens[i]]:
				tags[k] = 0
		else:
		# treat all unrecognized tokens as NNP
			emissions[tokens[i]] = {}
			emissions[tokens[i]]["NNP"] = 1.0
			tags["NNP"] = 0

	tags = list(tags.keys())

	# nested
	return(forward(tokens, tags))

def forward(tokens, tags):
	# build trellis of probabilities and indices of possible tags

	trellis = np.full((len(tokens), len(tags), 2), dtype=np.float64, fill_value=0)

	# first token
	curr_word = tokens[0]
	for j in range(len(tags)):
		curr_tag = tags[j]

		if curr_tag in emissions[curr_word]:				
			prev_tag = "START"

			if curr_tag in transitions:
				if prev_tag in transitions[curr_tag]:
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
							if prev_tag in transitions[curr_tag]:
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

def normalize_probabilities(sigma, delta):
	for word in emissions:
		c = 0
		for v in emissions[word].values():
			v += sigma
			c += v
		for k in emissions[word].keys():
			emissions[word][k] /= c

	for tag in transitions:
		c = 0
		for v in transitions[tag].values():
			v += delta
			c += v
		for k in transitions[tag].keys():
			transitions[tag][k] /= c

# global variables for laziness
# recall = {}
# precision = {}

emissions = {}       # "word/tag"      -> c("word/tag")
transitions = {}     # "tag_i/tag_j"   -> c("tag_i/tag_j")

main()
