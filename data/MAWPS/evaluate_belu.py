import json
import copy
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

def read_data_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def write_data_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


pairs_trained = read_data_json("train.json")
pairs_valided = read_data_json("valid.json")
pairs_tested = read_data_json("test.json")

reference_valid_list = []
reference_test_list = []
for i in range(len(pairs_valided)):
    reference_valid_list.append([pairs_valided[i][1]])
for i in range(len(pairs_tested)):
    reference_test_list.append([pairs_tested[i][1]])

output_valid_list = []
for i in range(len(pairs_valided)):
    output_valid_list.append(pairs_trained[pairs_valided[i][-1][0]][1])
print("valid_score:", corpus_bleu(reference_valid_list, output_valid_list))

output_test_list = []
for i in range(len(pairs_tested)):
    output_test_list.append(pairs_trained[pairs_tested[i][-1][0]][1])
print("test_score:", corpus_bleu(reference_test_list, output_test_list))

result_list = list(map(lambda x: json.loads(x), 
    open("Result/Result_test.txt", 'r').readlines()))
myoutput_test_list = []
for i in range(len(result_list)):
    temp = pairs_trained[pairs_tested[i][-1][0]]
    x = copy.deepcopy(pairs_tested[i][7])
    y = copy.deepcopy(temp[7])
    similary = len(set(x) & set(y)) / len(set(x) | set(y))
    if similary > 1:
        myoutput_test_list.append(temp[1])
    else:
#        output = result_list[i].split(" ")
#        if len(output) <= 10:
#            myoutput_test_list.append(temp[1])
#        else:
#        output[-1] = output[-1].replace("\n", "")
        output = result_list[i]["problem"][:-1]
        myoutput_test_list.append(output)
print("test_score:", corpus_bleu(reference_test_list, myoutput_test_list))
