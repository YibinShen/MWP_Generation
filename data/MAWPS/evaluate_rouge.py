import copy
import json
from rouge_metric import PyRouge

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
    reference_valid_list.append([" ".join(pairs_valided[i][1])])
for i in range(len(pairs_tested)):
    reference_test_list.append([" ".join(pairs_tested[i][1])])

output_valid_list = []
for i in range(len(pairs_valided)):
    output_valid_list.append(" ".join(pairs_trained[pairs_valided[i][-1][0]][1]))
output_test_list = []
for i in range(len(pairs_tested)):
    output_test_list.append(" ".join(pairs_trained[pairs_tested[i][-1][0]][1]))

rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
scores = rouge.evaluate(output_test_list, reference_test_list)
print("Knn:", scores)
print("-----------------------")

result_list = list(map(lambda x: json.loads(x), 
    open("Result/Result_test.txt", 'r').readlines()))
myoutput_test_list = []
for i in range(len(result_list)):
    temp = pairs_trained[pairs_tested[i][-1][0]]
    x = copy.deepcopy(pairs_tested[i][7])
    y = copy.deepcopy(temp[7])
    similary = len(set(x) & set(y)) / len(set(x) | set(y))
    if similary > 1:
        myoutput_test_list.append(" ".join(temp[1]))
    else:
#        output = result_list[i].split(" ")
#        if len(output) <= 10:
#            myoutput_test_list.append(temp[1])
#        else:
        output = result_list[i]["problem"][:-1]
        myoutput_test_list.append(" ".join(output))
rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
scores = rouge.evaluate(myoutput_test_list, reference_test_list)
print("output:",scores)
