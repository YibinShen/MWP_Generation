import random
import json
import copy
import re
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

PAD_token = 0


class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count, copy_nums):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "SOS", "EOS", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "SOS", "EOS", "NUM"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i


def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data


def load_mawps_data(filename):  # load the json data to list(dict()) for MAWPS
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    out_data = []
    for d in data:
        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[2:]
                out_data.append(temp)
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[:-2]
                out_data.append(temp)
                continue
    return out_data


def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        idx = d["id"]
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((idx, input_seq, out_seq, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums


def transfer_english_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = []
    generate_nums = {}
    copy_nums = 0
    idx = 1
    for d in data:
        nums = []
        input_seq = []
        seg = d["new_text"].strip().split(" ")
        equations = d["lEquations"]

        for s in seg:
            if s == "":
                continue
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                # if num[-2:] == ".0":
                #     num = num[:-2]
                # if "." in num and num[-1] == "0":
                #     num = num[:-1]
                nums.append(num.replace(",", ""))
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)
        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e not in "()+-*/":
                temp_eq += e
            elif temp_eq != "":
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq
                if len(count_eq) == 0:
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N"+str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        # def seg_and_tag(st):  # seg the equation and tag the num
        #     res = []
        #     pos_st = re.search(pattern, st)
        #     if pos_st:
        #         p_start = pos_st.start()
        #         p_end = pos_st.end()
        #         if p_start > 0:
        #             res += seg_and_tag(st[:p_start])
        #         st_num = st[p_start:p_end]
        #         if st_num[-2:] == ".0":
        #             st_num = st_num[:-2]
        #         if "." in st_num and st_num[-1] == "0":
        #             st_num = st_num[:-1]
        #         if nums.count(st_num) == 1:
        #             res.append("N"+str(nums.index(st_num)))
        #         else:
        #             res.append(st_num)
        #         if p_end < len(st):
        #             res += seg_and_tag(st[p_end:])
        #     else:
        #         for sst in st:
        #             res.append(sst)
        #     return res
        # out_seq = seg_and_tag(equations)

        # for s in out_seq:  # tag the num which is generated
        #     if s[0].isdigit() and s not in generate_nums and s not in nums:
        #         generate_nums.append(s)
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        if len(nums) != 0:
            pairs.append((str(idx), input_seq, eq_segs, nums, num_pos))
        idx += 1
        
    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res


def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    problem_lang = Lang()
    equation_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        problem_lang.add_sen_to_vocab(pair[1])
        problem_lang.add_sen_to_vocab(pair[7])
        equation_lang.add_sen_to_vocab(pair[4])
    
    problem_lang.build_input_lang(trim_min_count, copy_nums)
    if tree:
        equation_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        equation_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack =[]
        for word in pair[4]:
            temp_num = []
            flag_not = True
            if word not in equation_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[5]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[5]))])
        
        num_stack.reverse()
        keywords_cell = indexes_from_sentence(problem_lang, pair[7], True)
        problem_cell = indexes_from_sentence(problem_lang, pair[1], False)
        equation_cell = indexes_from_sentence(equation_lang, pair[4], tree)
        
        random.shuffle(pair[-2])
        for i in range(min(len(pair[-2]), 10)):
            prototype_cell = []
            prototype_pair = copy.deepcopy(pairs_trained[pair[-2][i]])
            prototype_keywords_cell = indexes_from_sentence(problem_lang, prototype_pair[7], True)
            prototype_problem_cell = indexes_from_sentence(problem_lang, prototype_pair[1], True)
            prototype_graph_cell = (prototype_pair[2], prototype_pair[3])
            prototype_cell.append([prototype_keywords_cell, prototype_problem_cell, prototype_graph_cell])
            
            train_pairs.append((pair[0], keywords_cell, len(keywords_cell), problem_cell, len(problem_cell),
                                equation_cell, len(equation_cell), prototype_cell, len(prototype_cell), 
                                pair[5], pair[6], num_stack))
    
    print('Indexed %d words in input language, %d words in output' % (problem_lang.n_words, equation_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        num_stack = []
        for word in pair[4]:
            temp_num = []
            flag_not = True
            if word not in equation_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[5]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[5]))])

        num_stack.reverse()
        keywords_cell = indexes_from_sentence(problem_lang, pair[7], True)
        problem_cell = indexes_from_sentence(problem_lang, pair[1], False)
        equation_cell = indexes_from_sentence(equation_lang, pair[4], tree)
        
        for i in range(1):
            prototype_cell = []
            prototype_pair = copy.deepcopy(pairs_trained[pair[-1][i]])
            prototype_keywords_cell = indexes_from_sentence(problem_lang, prototype_pair[7], True)
            prototype_problem_cell = indexes_from_sentence(problem_lang, prototype_pair[1], True)
            prototype_graph_cell = (prototype_pair[2], prototype_pair[3])
            prototype_cell.append([prototype_keywords_cell, prototype_problem_cell, prototype_graph_cell])
            
            test_pairs.append((pair[0], keywords_cell, len(keywords_cell), problem_cell, len(problem_cell),
                               equation_cell, len(equation_cell), prototype_cell, len(prototype_cell), 
                               pair[5], pair[6], num_stack))
    
    print('Number of testind data %d' % (len(test_pairs)))
    return problem_lang, equation_lang, train_pairs, test_pairs


def prepare_train_data(pairs_trained, problem_lang, equation_lang, tree=False):
    train_pairs = []
    for pair in pairs_trained:
        num_stack =[]
        for word in pair[4]:
            temp_num = []
            flag_not = True
            if word not in equation_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[5]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[5]))])
        
        num_stack.reverse()
        keywords_cell = indexes_from_sentence(problem_lang, pair[7], True)
        problem_cell = indexes_from_sentence(problem_lang, pair[1], False)
        equation_cell = indexes_from_sentence(equation_lang, pair[4], tree)
        
        random.shuffle(pair[-2])
        for i in range(min(len(pair[-2]), 10)):
            prototype_cell = []
            prototype_pair = copy.deepcopy(pairs_trained[pair[-2][i]])
            prototype_keywords_cell = indexes_from_sentence(problem_lang, prototype_pair[7], True)
            prototype_problem_cell = indexes_from_sentence(problem_lang, prototype_pair[1], True)
            prototype_graph_cell = (prototype_pair[2], prototype_pair[3])
            prototype_cell.append([prototype_keywords_cell, prototype_problem_cell, prototype_graph_cell])
            
            train_pairs.append((pair[0], keywords_cell, len(keywords_cell), problem_cell, len(problem_cell),
                                equation_cell, len(equation_cell), prototype_cell, len(prototype_cell), 
                                pair[5], pair[6], num_stack))

    return train_pairs


# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
    temp = copy.deepcopy(seq)
    temp += [PAD_token for _ in range(max_length - seq_len)]
    return temp


# prepare the batches
def prepare_train_batch(train_pairs, max_tokens, topk=5):
    pairs = copy.deepcopy(train_pairs)
    random.shuffle(pairs)  # shuffle the pairs
    keywords_lengths = []
    problem_lengths = []
    equation_lengths = []
    prototype_keywords_lengths = []
    prototype_problem_lengths = []
#    prototype_belong_indexes = []
    
    batches = []
    id_batches = []
    keywords_batches = []
    problem_batches = []
    equation_batches = []
    prototype_keywords_batches = []
    prototype_problem_batches = []
    prototype_graph1_batches = []
    prototype_graph2_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
#    pos = 0
    start = 0
    tokens_len = 0
    for i in range(len(pairs)):
        tokens_len += len(pairs[i][7][0][1])
        if tokens_len > max_tokens:
            batches.append(pairs[start:i])
            start = i
            tokens_len = 0
    batches.append(pairs[start:(i+1)])
#    while pos + batch_size < len(pairs):
#        batches.append(pairs[pos:pos+batch_size])
#        pos += batch_size
#    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: len(tp[7][0][1]), reverse=True)
        problem_length = []
        equation_length = []
        prototype_problem_length = []
        prototype_keywords = []
        prototype_problem= []
        prototype_graph1 = []
        prototype_graph2 = []
#        prototype_belong_index = []
        for batch_id, (_, _, i, _, j, _, k, prototype_pairs, l, _, _, _) in enumerate(batch):
            problem_length.append(j)
            equation_length.append(k)
            for pair in prototype_pairs:
                prototype_keywords.append(pair[0])
                prototype_problem.append(pair[1])
                prototype_graph1.append(pair[2][0])
                prototype_graph2.append(pair[2][1])
                prototype_problem_length.append(len(pair[1]))
#                prototype_belong_index.append(batch_id)
#        prototype_sorted_index = np.argsort(-np.array(prototype_problem_length))
#        prototype_keywords = np.array(prototype_keywords)[prototype_sorted_index].tolist()
#        prototype_problem = np.array(prototype_problem)[prototype_sorted_index].tolist()
#        prototype_graph1 = np.array(prototype_graph1)[prototype_sorted_index].tolist()
#        prototype_graph2 = np.array(prototype_graph2)[prototype_sorted_index].tolist()
#        prototype_problem_length = np.array(prototype_problem_length)[prototype_sorted_index].tolist()
#        prototype_belong_index = np.array(prototype_belong_index)[prototype_sorted_index].tolist()
        
        keywords_lengths.append([topk] * len(batch))
        problem_lengths.append(problem_length)
        equation_lengths.append(equation_length)
        prototype_keywords_lengths.append([topk] * len(prototype_keywords))
        prototype_problem_lengths.append(prototype_problem_length)
        problem_len_max = max(problem_length)
        equation_len_max = max(equation_length)
        prototype_problem_len_max = max(prototype_problem_length)
        
        id_batch = []
        problem_batch = []
        keywords_batch = []
        equation_batch = []
        prototype_keywords_batch = []
        prototype_problem_batch = []
        prototype_graph1_batch = []
        prototype_graph2_batch = []
        num_size_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        for idx, i, li, j, lj, k, lk, _, _, num_size, num_pos, num_stack in batch:
            id_batch.append(idx)
            keywords_batch.append(pad_seq(i, li, topk))
            problem_batch.append(pad_seq(j, lj, problem_len_max))
            equation_batch.append(pad_seq(k, lk, equation_len_max))
#            prototype_keywords_batch.append(pad_seq(k, lk, topk))
#            prototype_problem_batch.append(pad_seq(l, ll, prototype_problem_len_max))
#            prototype_graph_batch.append(graph)
            num_size_batch.append(len(num_size))
            num_stack_batch.append(copy.deepcopy(num_stack))
            num_pos_batch.append(copy.deepcopy(num_pos))
        
        for i in range(len(prototype_keywords)):
#            prototype_keywords_batch.append(pad_seq(prototype_keywords[i], 
#                                            len(prototype_keywords[i]), topk))
            prototype_problem_batch.append(pad_seq(prototype_problem[i], 
                                           prototype_problem_length[i], prototype_problem_len_max))
            prototype_graph1_pad, prototype_graph2_pad = get_graph(prototype_graph1[i], 
                                                                   prototype_graph2[i],
                                                                   prototype_problem_len_max)
            prototype_graph1_batch.append(prototype_graph1_pad)
            prototype_graph2_batch.append(prototype_graph2_pad)
        prototype_keywords_batch = prototype_keywords
        
        id_batches.append(id_batch)
        keywords_batches.append(keywords_batch)
        problem_batches.append(problem_batch)
        equation_batches.append(equation_batch)
        prototype_keywords_batches.append(prototype_keywords_batch)
        prototype_problem_batches.append(prototype_problem_batch)
        prototype_graph1_batches.append(prototype_graph1_batch)
        prototype_graph2_batches.append(prototype_graph2_batch)
#        prototype_belong_indexes.append(prototype_belong_index)
        num_size_batches.append(num_size_batch)
        num_pos_batches.append(num_pos_batch)
        num_stack_batches.append(num_stack_batch)
    
    return id_batches, keywords_batches, keywords_lengths, problem_batches, problem_lengths, \
        equation_batches, equation_lengths, prototype_keywords_batches, prototype_keywords_lengths, \
        prototype_problem_batches, prototype_problem_lengths, prototype_graph1_batches, prototype_graph2_batches, \
        num_size_batches, num_pos_batches, num_stack_batches


def get_graph(graph_src, graph_tgt, max_len):
    diag_ele = [1] * len(graph_tgt) + [0] * (max_len - len(graph_tgt))
    graph1 = np.diag([1]*max_len) + np.diag(diag_ele[1:], 1) + np.diag(diag_ele[1:], -1)
    graph2 = np.diag([1]*max_len) + np.diag(diag_ele[1:], 1) + np.diag(diag_ele[1:], -1)
    for i in range(len(graph_tgt)):
        if graph_tgt[i] != -1:
            graph1[graph_src[i], graph_tgt[i]] = 1
            graph1[graph_tgt[i], graph_src[i]] = 1
            graph2[graph_tgt[i], graph_src[i]] = 1
    return graph1.tolist(), graph2.tolist()


def prepare_test_batch(test_pairs, batch_size, topk=5):
    pairs = copy.deepcopy(test_pairs)
    keywords_lengths = []
    problem_lengths = []
    equation_lengths = []
    prototype_keywords_lengths = []
    prototype_problem_lengths = []
#    prototype_belong_indexes = []
    
    batches = []
    id_batches = []
    keywords_batches = []
    problem_batches = []
    equation_batches = []
    prototype_keywords_batches = []
    prototype_problem_batches = []
    prototype_graph1_batches = []
    prototype_graph2_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    pos = 0
#    start = 0
#    tokens_len = 0
#    for i in range(len(pairs)):
#        tokens_len += pairs[i][8]
#        if tokens_len > max_tokens:
#            batches.append(pairs[start:i])
#            start = i
#            tokens_len = 0
#    batches.append(pairs[start:(i+1)])
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: len(tp[7][0][1]), reverse=True)
        problem_length = []
        equation_length = []
        prototype_problem_length = []
        prototype_keywords = []
        prototype_problem= []
        prototype_graph1 = []
        prototype_graph2 = []
#        prototype_belong_index = []
        for batch_id, (_, _, i, _, j, _, k, prototype_pairs, l, _, _, _) in enumerate(batch):
            problem_length.append(j)
            equation_length.append(k)
            for pair in prototype_pairs:
                prototype_keywords.append(pair[0])
                prototype_problem.append(pair[1])
                prototype_graph1.append(pair[2][0])
                prototype_graph2.append(pair[2][1])
                prototype_problem_length.append(len(pair[1]))
#                prototype_belong_index.append(batch_id)
#        prototype_sorted_index = np.argsort(-np.array(prototype_problem_length))
#        prototype_keywords = np.array(prototype_keywords)[prototype_sorted_index].tolist()
#        prototype_problem = np.array(prototype_problem)[prototype_sorted_index].tolist()
#        prototype_graph1 = np.array(prototype_graph1)[prototype_sorted_index].tolist()
#        prototype_graph2 = np.array(prototype_graph2)[prototype_sorted_index].tolist()
#        prototype_problem_length = np.array(prototype_problem_length)[prototype_sorted_index].tolist()
#        prototype_belong_index = np.array(prototype_belong_index)[prototype_sorted_index].tolist()
        
        keywords_lengths.append([topk] * len(batch))
        problem_lengths.append(problem_length)
        equation_lengths.append(equation_length)
        prototype_keywords_lengths.append([topk] * len(prototype_keywords))
        prototype_problem_lengths.append(prototype_problem_length)
        problem_len_max = max(problem_length)
        equation_len_max = max(equation_length)
        prototype_problem_len_max = max(prototype_problem_length)
        
        id_batch = []
        problem_batch = []
        keywords_batch = []
        equation_batch = []
        prototype_keywords_batch = []
        prototype_problem_batch = []
        prototype_graph1_batch = []
        prototype_graph2_batch = []
        num_size_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        for idx, i, li, j, lj, k, lk, _, _, num_size, num_pos, num_stack in batch:
            id_batch.append(idx)
            keywords_batch.append(pad_seq(i, li, topk))
            problem_batch.append(pad_seq(j, lj, problem_len_max))
            equation_batch.append(pad_seq(k, lk, equation_len_max))
#            prototype_keywords_batch.append(pad_seq(k, lk, topk))
#            prototype_problem_batch.append(pad_seq(l, ll, prototype_problem_len_max))
#            prototype_graph_batch.append(graph)
            num_size_batch.append(len(num_size))
            num_stack_batch.append(copy.deepcopy(num_stack))
            num_pos_batch.append(copy.deepcopy(num_pos))
        
        for i in range(len(prototype_keywords)):
#            prototype_keywords_batch.append(pad_seq(prototype_keywords[i], 
#                                            len(prototype_keywords[i]), topk))
            prototype_problem_batch.append(pad_seq(prototype_problem[i], 
                                           prototype_problem_length[i], prototype_problem_len_max))
            prototype_graph1_pad, prototype_graph2_pad = get_graph(prototype_graph1[i], 
                                                                   prototype_graph2[i],
                                                                   prototype_problem_len_max)
            prototype_graph1_batch.append(prototype_graph1_pad)
            prototype_graph2_batch.append(prototype_graph2_pad)
        prototype_keywords_batch = prototype_keywords
        
        id_batches.append(id_batch)
        keywords_batches.append(keywords_batch)
        problem_batches.append(problem_batch)
        equation_batches.append(equation_batch)
        prototype_keywords_batches.append(prototype_keywords_batch)
        prototype_problem_batches.append(prototype_problem_batch)
        prototype_graph1_batches.append(prototype_graph1_batch)
        prototype_graph2_batches.append(prototype_graph2_batch)
#        prototype_belong_indexes.append(prototype_belong_index)
        num_size_batches.append(num_size_batch)
        num_pos_batches.append(num_pos_batch)
        num_stack_batches.append(num_stack_batch)
    
    return id_batches, keywords_batches, keywords_lengths, problem_batches, problem_lengths, \
        equation_batches, equation_lengths, prototype_keywords_batches, prototype_keywords_lengths, \
        prototype_problem_batches, prototype_problem_lengths, prototype_graph1_batches, prototype_graph2_batches, \
        num_size_batches, num_pos_batches, num_stack_batches


def generate_english_keywords_valid(pairs_train, pairs_valid, pairs_test, 
                                    topk=5, alpha=0.25, beta=0.1):
    train_document = []
    valid_document = []
    test_document = []
    for i in range(len(pairs_train)):
        train_document.append(" ".join(pairs_train[i][1]))
    for i in range(len(pairs_valid)):
        valid_document.append(" ".join(pairs_valid[i][1]))
    for i in range(len(pairs_test)):
        test_document.append(" ".join(pairs_test[i][1]))
    
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    tfidf_model = tfidf_vectorizer.fit(train_document)
    sparse_result = tfidf_model.transform(train_document + valid_document + test_document)
    result = sparse_result.toarray()
    train_result = result[:len(pairs_train)]
    valid_result = result[len(pairs_train):(len(pairs_train)+len(pairs_valid))]
    test_result = result[-len(pairs_test):]
    words = tfidf_model.get_feature_names()
    
    # extract keywords
    for i in range(len(train_result)):
        x = copy.deepcopy(train_result[i])
        word_index = np.argsort(-x)[:topk]
        keywords = []
        for temp in word_index:
            if x[temp] > 0:
                keywords.append(words[temp])
            else:
                keywords.append("PAD")
        pairs_train[i].append(keywords)
    
    for i in range(len(valid_result)):
        x = copy.deepcopy(valid_result[i])
        word_index = np.argsort(-x)[:topk]
        keywords = []
        for temp in word_index:
            if x[temp] > 0:
                keywords.append(words[temp])
            else:
                keywords.append("PAD")
        pairs_valid[i].append(keywords)
    
    for i in range(len(test_result)):
        x = copy.deepcopy(test_result[i])
        word_index = np.argsort(-x)[:topk]
        keywords = []
        for temp in word_index:
            if x[temp] > 0:
                keywords.append(words[temp])
            else:
                keywords.append("PAD")
        pairs_test[i].append(keywords)
    
    for i in range(len(train_result)):
        x = copy.deepcopy(pairs_train[i][7])
        id_array = np.zeros(len(pairs_train))
        len_array = np.zeros(len(pairs_train))
        for j in range(len(train_result)):
            y = copy.deepcopy(pairs_train[j][7])
            id_array[j] = len(set(x) & set(y)) / len(set(x) | set(y))
            len_array[j] = len(pairs_train[j][1]) / len(pairs_train[i][1])
        id_index = np.where((id_array>=alpha) & (id_array<=(1-alpha)) & \
                            (len_array>=(1-beta)) & (len_array<=(1+beta)))[0].tolist()
#        id_index = np.where((id_array>=alpha) & (id_array<=(1-alpha)))[0].tolist()
        if len(id_index) == 0:
            id_index += [i]
        pairs_train[i].append(id_index)
        pairs_train[i].append([i])
    
    for i in range(len(valid_result)):
        x = copy.deepcopy(pairs_valid[i][7])
        id_array = np.zeros(len(pairs_train))
        for j in range(len(train_result)):
            y = copy.deepcopy(pairs_train[j][7])
            id_array[j] = len(set(x) & set(y)) / len(set(x) | set(y))
#        id_index = np.where((id_array>=alpha) & (id_array<=(1-alpha)))[0].tolist()
#        id_index = np.where((id_array>=2*alpha))[0].tolist()
        id_closest_index = np.argsort(-id_array)[0:1].tolist()
        id_array[id_array>(1-alpha)] = 0
        id_index = np.argsort(-id_array)[0:1].tolist()
        pairs_valid[i].append(id_index)
        pairs_valid[i].append(id_closest_index)
    
    for i in range(len(test_result)):
        x = copy.deepcopy(pairs_test[i][7])
        id_array = np.zeros(len(pairs_train))
        for j in range(len(train_result)):
            y = copy.deepcopy(pairs_train[j][7])
            id_array[j] = len(set(x) & set(y)) / len(set(x) | set(y))
#        id_index = np.where((id_array>=alpha) & (id_array<=(1-alpha)))[0].tolist()
#        id_index = np.where((id_array>=2*alpha))[0].tolist()
        id_closest_index = np.argsort(-id_array)[0:1].tolist()
        id_array[id_array>(1-alpha)] = 0
        id_index = np.argsort(-id_array)[0:1].tolist()
        pairs_test[i].append(id_index)
        pairs_test[i].append(id_closest_index)
    
    return pairs_train, pairs_valid, pairs_test


def generate_chinese_keywords_valid(pairs_train, pairs_valid, pairs_test, 
                                    topk=5, alpha=0.25, beta=0.1):
    train_document = []
    valid_document = []
    test_document = []
    for i in range(len(pairs_train)):
        train_document.append(" ".join(pairs_train[i][1]))
    for i in range(len(pairs_valid)):
        valid_document.append(" ".join(pairs_valid[i][1]))
    for i in range(len(pairs_test)):
        test_document.append(" ".join(pairs_test[i][1]))
    
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf_model = tfidf_vectorizer.fit(train_document)
    sparse_result = tfidf_model.transform(train_document + valid_document + test_document)
    result = sparse_result.toarray()
    train_result = result[:len(pairs_train)]
    valid_result = result[len(pairs_train):(len(pairs_train)+len(pairs_valid))]
    test_result = result[-len(pairs_test):]
    words = tfidf_model.get_feature_names()
    
    # extract keywords
    for i in range(len(train_result)):
        x = copy.deepcopy(train_result[i])
        word_index = np.argsort(-x)[:topk]
        keywords = []
        for temp in word_index:
            if x[temp] > 0:
                keywords.append(words[temp])
            else:
                keywords.append("PAD")
        pairs_train[i].append(keywords)
    
    for i in range(len(valid_result)):
        x = copy.deepcopy(valid_result[i])
        word_index = np.argsort(-x)[:topk]
        keywords = []
        for temp in word_index:
            if x[temp] > 0:
                keywords.append(words[temp])
            else:
                keywords.append("PAD")
        pairs_valid[i].append(keywords)
    
    for i in range(len(test_result)):
        x = copy.deepcopy(test_result[i])
        word_index = np.argsort(-x)[:topk]
        keywords = []
        for temp in word_index:
            if x[temp] > 0:
                keywords.append(words[temp])
            else:
                keywords.append("PAD")
        pairs_test[i].append(keywords)
    
    for i in range(len(train_result)):
        x = copy.deepcopy(pairs_train[i][7])
        id_array = np.zeros(len(pairs_train))
        len_array = np.zeros(len(pairs_train))
        for j in range(len(train_result)):
            y = copy.deepcopy(pairs_train[j][7])
            id_array[j] = len(set(x) & set(y)) / len(set(x) | set(y))
            len_array[j] = len(pairs_train[j][1]) / len(pairs_train[i][1])
        id_index = np.where((id_array>=alpha) & (id_array<=(1-alpha)) & \
                            (len_array>=(1-beta)) & (len_array<=(1+beta)))[0].tolist()
#        id_index = np.where((id_array>=alpha) & (id_array<=(1-alpha)))[0].tolist()
        if len(id_index) == 0:
            id_index += [i]
        pairs_train[i].append(id_index)
        pairs_train[i].append([i])
    
    for i in range(len(valid_result)):
        x = copy.deepcopy(pairs_valid[i][7])
        id_array = np.zeros(len(pairs_train))
        for j in range(len(train_result)):
            y = copy.deepcopy(pairs_train[j][7])
            id_array[j] = len(set(x) & set(y)) / len(set(x) | set(y))
        id_index = np.where((id_array>=alpha) & (id_array<=(1-alpha)))[0].tolist()
#        id_index = np.where((id_array>=2*alpha))[0].tolist()
        id_closest_index = np.argsort(-id_array)[0:1].tolist()
        if len(id_index) == 0:
            id_index += id_closest_index
        pairs_valid[i].append(id_index)
        pairs_valid[i].append(id_closest_index)
    
    for i in range(len(test_result)):
        x = copy.deepcopy(pairs_test[i][7])
        id_array = np.zeros(len(pairs_train))
        for j in range(len(train_result)):
            y = copy.deepcopy(pairs_train[j][7])
            id_array[j] = len(set(x) & set(y)) / len(set(x) | set(y))
        id_index = np.where((id_array>=alpha) & (id_array<=(1-alpha)))[0].tolist()
#        id_index = np.where((id_array>=2*alpha))[0].tolist()
        id_closest_index = np.argsort(-id_array)[0:1].tolist()
        if len(id_index) == 0:
            id_index += id_closest_index
        pairs_test[i].append(id_index)
        pairs_test[i].append(id_closest_index)
    
    return pairs_train, pairs_valid, pairs_test