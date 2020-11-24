# coding: utf-8
import os
import time
import json
import numpy as np
import torch.optim
import torch.nn as nn
from src.models import *
from src.train_and_evaluate import *
from src.expressions_transfer import *
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from pyltp import Postagger,Parser
#LTP_DATA_DIR="../LTP/ltp_data_v3.4.0"
#pos_model_path = os.path.join(LTP_DATA_DIR, "pos.model")
#par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
#postagger = Postagger()
#postagger.load(pos_model_path)
#parser = Parser()
#parser.load(par_model_path)

max_tokens = 4096
embedding_size = 512
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
topk = 10
dropout = 0.5

def read_data_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def write_data_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def write_data_headlines_json(data, filename):
    with open(filename, 'w') as f:
        for p in data:
            json.dump(p, f, ensure_ascii=False)
            f.write("\n")
    f.close()

def generate_processed_data():
    data = load_raw_data("data/Math23K/Math23K.json")
    pairs, generate_nums, copy_nums = transfer_num(data)
    data_list = []
    data_problem_list = []
    for i in range(len(pairs)):
        temp = copy.deepcopy(pairs[i])
        temp_mask_text = " ".join(temp[1])
        if temp_mask_text in data_problem_list:
            continue
        data_list.append(data[i])
        data_problem_list.append(temp_mask_text)
    
    write_data_json(data_list, "data/Math23K/Math23K_filter.json")
    pairs, generate_nums, copy_nums = transfer_num(data_list)

    temp_pairs = []
    for p in pairs:
        postags = postagger.postag(p[1])
        postags = " ".join(postags).split(" ")
        arcs = parser.parse(p[1], postags)
        parse_tgt = [arc.head-1 for arc in arcs]
        parse_src = np.arange(len(parse_tgt)).tolist()
        if len(parse_tgt) != len(p[1]):
            print(p[0])
        
        temp_pairs.append((p[0], p[1], parse_src, parse_tgt,
                           from_infix_to_prefix(p[2]), p[3], p[4]))
    
#    for p in pairs:
#        temp_pairs.append((p[0], p[1], from_infix_to_prefix(p[2]), p[3], p[4]))

    write_data_json(temp_pairs, "data/Math23K/Math23K_processed.json")

def generate_valid5_data():
    pairs = read_data_json("data/Math23K/Math23K_processed.json")
    fold_size = int(len(pairs) * 0.2)
    fold_pairs = []
    for split_fold in range(4):
        fold_start = fold_size * split_fold
        fold_end = fold_size * (split_fold + 1)
        fold_pairs.append(pairs[fold_start:fold_end])
    fold_pairs.append(pairs[(fold_size * 4):])

    for fold in range(5):
        pairs_tested = []
        pairs_trained = []
        for fold_t in range(5):
            if fold_t == fold:
                pairs_tested += fold_pairs[fold_t]
            else:
                pairs_trained += fold_pairs[fold_t]
        
        pairs_trained, pairs_tested = generate_chinese_keywords(copy.deepcopy(pairs_trained), copy.deepcopy(pairs_tested), topk)
        write_data_json(pairs_trained, "data/Math23K/train_"+str(fold)+".json")
        write_data_json(pairs_tested, "data/Math23K/test_"+str(fold)+".json")
        print(fold)

def generate_train_test_data():
    pairs = read_data_json("data/Math23K/Math23K_processed.json")
    pairs_test = pairs[:int(len(pairs) * 0.2)]
    pairs_valid = pairs[int(len(pairs) * 0.2):int(len(pairs) * 0.4)]
    pairs_train = pairs[int(len(pairs) * 0.4):]
    
    pairs_trained, pairs_valided, pairs_tested = generate_chinese_keywords_valid(copy.deepcopy(pairs_train),
                                                                                 copy.deepcopy(pairs_valid),
                                                                                 copy.deepcopy(pairs_test), topk)
    write_data_json(pairs_trained, "data/Math23K/train.json")
    write_data_json(pairs_valided, "data/Math23K/valid.json")
    write_data_json(pairs_tested, "data/Math23K/test.json")


def train():
    data = read_data_json("data/Math23K/Math23K_filter.json")
    pairs, generate_nums, copy_nums = transfer_num(data)

    pairs_trained = read_data_json("data/Math23K/train.json")
    pairs_valided = read_data_json("data/Math23K/valid.json")
    pairs_tested = read_data_json("data/Math23K/test.json")
    
    reference_valid_list = []
    reference_test_list = []
    for i in range(len(pairs_valided)):
        reference_valid_list.append([pairs_valided[i][1]])
    for i in range(len(pairs_tested)):
        reference_test_list.append([pairs_tested[i][1]])
    f = open("data/Math23K/reference_valid.txt", 'w')
    for i in range(len(reference_valid_list)):
        f.write(" ".join(reference_valid_list[i][0]))
        f.write("\n")
    f.close()
    f = open("data/Math23K/reference_test.txt", 'w')
    for i in range(len(reference_test_list)):
        f.write(" ".join(reference_test_list[i][0]))
        f.write("\n")
    f.close()
    
    output_valid_list = []
    for i in range(len(pairs_valided)):
        output_valid_list.append(pairs_trained[pairs_valided[i][-1][0]][1])
    f = open("data/Math23K/output_valid_knn.txt", 'w')
    for i in range(len(output_valid_list)):
        f.write(" ".join(output_valid_list[i]))
        f.write("\n")
    f.close()
    print("valid_score:", corpus_bleu(reference_valid_list, output_valid_list))
    
    output_test_list = []
    for i in range(len(pairs_tested)):
        output_test_list.append(pairs_trained[pairs_tested[i][-1][0]][1])
    f = open("data/Math23K/output_test_knn.txt", 'w')
    for i in range(len(output_test_list)):
        f.write(" ".join(output_test_list[i]))
        f.write("\n")
    f.close()
    print("test_score:", corpus_bleu(reference_test_list, output_test_list))
    
    problem_lang, equation_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_valided, 1, generate_nums, copy_nums, tree=True)
    json.dump(problem_lang.word2index, open("data/Math23K/Dict/problem_dict.json", 'w'), indent=4)
    json.dump(equation_lang.word2index, open("data/Math23K/Dict/equation_dict.json", 'w'), indent=4)

    # Initialize models
    problem_embed_model = nn.Embedding(problem_lang.n_words, embedding_size, padding_idx=0)
    
    keywords_encoder = EncoderKeywords(input_size=problem_lang.n_words, embed_model=problem_embed_model, 
                                       embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
    prototype_encoder = EncoderPrototype(input_size1=problem_lang.n_words, input_size2=problem_lang.n_words, embed_model=problem_embed_model,
                                         embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
    decoder = AttnDecoderRNN(hidden_size=hidden_size, embedding_size=embedding_size,
                             input_size=problem_lang.n_words, output_size=problem_lang.n_words, n_layers=n_layers, dropout=dropout)
    predict = Prediction(hidden_size=hidden_size, op_nums=equation_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums), dropout=dropout)
    generate = GenerateNode(hidden_size=hidden_size, op_nums=equation_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size, dropout=dropout)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size, dropout=dropout)

#    # the embedding layer is  only for generated number embeddings, operators, and paddings
    keywords_encoder_optimizer = torch.optim.Adam(keywords_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    prototype_encoder_optimizer = torch.optim.Adam(prototype_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    keywords_encoder_scheduler = torch.optim.lr_scheduler.StepLR(keywords_encoder_optimizer, step_size=20, gamma=0.5)
    prototype_encoder_scheduler = torch.optim.lr_scheduler.StepLR(prototype_encoder_optimizer, step_size=20, gamma=0.5)
    decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=20, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)
    
    # Move models to GPU
    if USE_CUDA:
        keywords_encoder.cuda()
        prototype_encoder.cuda()
        decoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()
    
    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(equation_lang.word2index[num])
    
    max_score = 0.0
    max_epoch = 0
    
    for epoch in range(n_epochs):
        train_pairs = prepare_train_data(pairs_trained, problem_lang, equation_lang, tree=True)
        loss_total = 0
        id_batches, keywords_batches, keywords_lengths, problem_batches, problem_lengths, \
            equation_batches, equation_lengths, prototype_keywords_batches, prototype_keywords_lengths, \
            prototype_problem_batches, prototype_problem_lengths, prototype_graph1_batches, prototype_graph2_batches, \
            num_size_batches, num_pos_batches, num_stack_batches = prepare_train_batch(train_pairs, max_tokens, topk)
        
        print("epoch:", epoch + 1)
        start = time.time()
        for idx in range(len(problem_lengths)):
            loss = train_problem(
                    keywords_batches[idx], keywords_lengths[idx], problem_batches[idx], problem_lengths[idx], 
                    equation_batches[idx], equation_lengths[idx], prototype_keywords_batches[idx], prototype_keywords_lengths[idx], 
                    prototype_problem_batches[idx], prototype_problem_lengths[idx], 
                    prototype_graph1_batches[idx], prototype_graph2_batches[idx], 
                    num_size_batches[idx], num_pos_batches[idx], num_stack_batches[idx], generate_num_ids, 
                    keywords_encoder, prototype_encoder, decoder, predict, generate, merge, 
                    keywords_encoder_optimizer, prototype_encoder_optimizer, decoder_optimizer, 
                    predict_optimizer, generate_optimizer, merge_optimizer, 
                    problem_lang, equation_lang, beam_size=10, use_teacher_forcing=1, lenpen1=1)
            
            loss_total += loss
            
        print("loss:", loss_total / len(problem_lengths))
        print("training time", time_since(time.time() - start))
        print("--------------------------------")
        if epoch % 10 == 0 or epoch > 30:
            start = time.time()
            f = open("data/Math23K/Result/Result_test.txt", 'w')
            id_batches, keywords_batches, keywords_lengths, problem_batches, problem_lengths, \
                equation_batches, equation_lengths, prototype_keywords_batches, prototype_keywords_lengths, \
                prototype_problem_batches, prototype_problem_lengths, prototype_graph1_batches, prototype_graph2_batches, \
                num_size_batches, num_pos_batches, num_stack_batches = prepare_test_batch(test_pairs, 1, topk)
            
            result_list = []
            for idx in range(len(problem_lengths)):
                result_dict = dict()
                problem_res, equation_res = evaluate_problem(keywords_batches[idx], keywords_lengths[idx], 
                                                             prototype_keywords_batches[idx], prototype_keywords_lengths[idx], 
                                                             prototype_problem_batches[idx], prototype_problem_lengths[idx], 
                                                             prototype_graph1_batches[idx], prototype_graph2_batches[idx],
                                                             generate_num_ids, copy_nums, 
                                                             keywords_encoder, prototype_encoder, 
                                                             decoder, predict, generate, merge, 
                                                             problem_lang, equation_lang, beam_size=10, 
                                                             max_length1=120, lenpen1=1)
                problem_result = out_problem_list(problem_res.all_output2, problem_lang)
                if equation_res != 0:
                    equation_result = out_problem_list(equation_res.out, equation_lang)
                    equation_score = equation_res.score
                else:
                    equation_result = None
                    equation_score = None
                result_dict["id"] = id_batches[idx]
                result_dict["problem"] = problem_result
                result_list.append(problem_result)
                result_dict["equation"] = equation_result
                result_dict["problem_score"] = problem_res.score
                result_dict["equation_score"] = equation_score
                json.dump(result_dict, f)
                f.write("\n")
            f.close()
            print("testing time", time_since(time.time() - start))
            print("------------------------------------------------------")
            
            torch.save(keywords_encoder.state_dict(), "data/Math23K/Models/keywords_encoder_"+str(epoch))
            torch.save(prototype_encoder.state_dict(), "data/Math23K/Models/prototype_encoder_"+str(epoch))
            torch.save(decoder.state_dict(), "data/Math23K/Models/decoder_"+str(epoch))
            torch.save(predict.state_dict(), "data/Math23K/Models/predict_"+str(epoch))
            torch.save(generate.state_dict(), "data/Math23K/Models/generate_"+str(epoch))
            torch.save(merge.state_dict(), "data/Math23K/Models/merge_"+str(epoch))
            
            myoutput_test_list = []
            for i in range(len(result_list)):
                temp = pairs_trained[pairs_valided[i][-1][0]]
                x = copy.deepcopy(pairs_valided[i][7])
                y = copy.deepcopy(temp[7])
                similary = len(set(x) & set(y)) / len(set(x) | set(y))
                if similary >= 1:
                    myoutput_test_list.append(temp[1])
                else:
                    myoutput_test_list.append(result_list[i][:-1])
            
            score = corpus_bleu(reference_valid_list, myoutput_test_list)
            print("my_score:", score)
            f = open("data/Math23K/output_valid.txt", 'w')
            for i in range(len(myoutput_test_list)):
                f.write(" ".join(myoutput_test_list[i]))
                f.write("\n")
            f.close()
            
            if score > max_score:
                max_score = score
                max_epoch = epoch
        
        if epoch - max_epoch > 5 and epoch > 30:
            print("max_epoch:", max_epoch)
            print("max_score:", max_score)
            break
        
        keywords_encoder_scheduler.step()
        prototype_encoder_scheduler.step()
        decoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()


def test(epoch):
    data = read_data_json("data/Math23K/Math23K_filter.json")
    pairs, generate_nums, copy_nums = transfer_num(data)
    
    pairs_trained = read_data_json("data/Math23K/train.json")
    pairs_tested = read_data_json("data/Math23K/test.json")
    
    reference_test_list = []
    for i in range(len(pairs_tested)):
        reference_test_list.append([pairs_tested[i][1]])
    
    problem_lang, equation_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 1, generate_nums, copy_nums, tree=True)
    
    # Initialize models
    problem_embed_model = nn.Embedding(problem_lang.n_words, embedding_size, padding_idx=0)
    
    keywords_encoder = EncoderKeywords(input_size=problem_lang.n_words, embed_model=problem_embed_model, 
                                       embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers)
    prototype_encoder = EncoderPrototype(input_size1=problem_lang.n_words, input_size2=problem_lang.n_words, embed_model=problem_embed_model,
                                         embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers)
    decoder = AttnDecoderRNN(hidden_size=hidden_size, embedding_size=embedding_size,
                             input_size=problem_lang.n_words, output_size=problem_lang.n_words, n_layers=n_layers)
    predict = Prediction(hidden_size=hidden_size, op_nums=equation_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=equation_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
    
    keywords_encoder.load_state_dict(torch.load("data/Math23K/Models/keywords_encoder_"+str(epoch)))
    prototype_encoder.load_state_dict(torch.load("data/Math23K/Models/prototype_encoder_"+str(epoch)))
    decoder.load_state_dict(torch.load("data/Math23K/Models/decoder_"+str(epoch)))
    predict.load_state_dict(torch.load("data/Math23K/Models/predict_"+str(epoch)))
    generate.load_state_dict(torch.load("data/Math23K/Models/generate_"+str(epoch)))
    merge.load_state_dict(torch.load("data/Math23K/Models/merge_"+str(epoch)))
    
    # Move models to GPU
    if USE_CUDA:
        keywords_encoder.cuda()
        prototype_encoder.cuda()
        decoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()
    
    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(equation_lang.word2index[num])
    
    start = time.time()
    f = open("data/Math23K/Result/Result_test.txt", 'w')
    id_batches, keywords_batches, keywords_lengths, problem_batches, problem_lengths, \
        equation_batches, equation_lengths, prototype_keywords_batches, prototype_keywords_lengths, \
        prototype_problem_batches, prototype_problem_lengths, prototype_graph1_batches, prototype_graph2_batches, \
        num_size_batches, num_pos_batches, num_stack_batches = prepare_test_batch(test_pairs, 1, topk)
    
    result_list = []
    for idx in range(len(problem_lengths)):
        result_dict = dict()
        problem_res, equation_res = evaluate_problem(keywords_batches[idx], keywords_lengths[idx], 
                                                     prototype_keywords_batches[idx], prototype_keywords_lengths[idx], 
                                                     prototype_problem_batches[idx], prototype_problem_lengths[idx], 
                                                     prototype_graph1_batches[idx], prototype_graph2_batches[idx],
                                                     generate_num_ids, copy_nums, 
                                                     keywords_encoder, prototype_encoder, 
                                                     decoder, predict, generate, merge, 
                                                     problem_lang, equation_lang, beam_size=10, 
                                                     max_length1=120, lenpen1=1)
        problem_result = out_problem_list(problem_res.all_output2, problem_lang)
        if equation_res != 0:
            equation_result = out_problem_list(equation_res.out, equation_lang)
            equation_score = equation_res.score
        else:
            equation_result = None
            equation_score = None
        result_dict["id"] = id_batches[idx]
        result_dict["problem"] = problem_result
        result_list.append(problem_result)
        result_dict["equation"] = equation_result
        result_dict["problem_score"] = problem_res.score
        result_dict["equation_score"] = equation_score
        json.dump(result_dict, f)
        f.write("\n")
    f.close()
    print("testing time", time_since(time.time() - start))
    print("------------------------------------------------------")
    
    myoutput_test_list = []
    for i in range(len(result_list)):
        temp = pairs_trained[pairs_tested[i][-1][0]]
        x = copy.deepcopy(pairs_tested[i][7])
        y = copy.deepcopy(temp[7])
        similary = len(set(x) & set(y)) / len(set(x) | set(y))
        if similary >= 1:
            myoutput_test_list.append(temp[1])
        else:
            myoutput_test_list.append(result_list[i][:-1])
    
    score = corpus_bleu(reference_test_list, myoutput_test_list)
    print("my_score:", score)
    f = open("data/Math23K/output_test.txt", 'w')
    for i in range(len(myoutput_test_list)):
        f.write(" ".join(myoutput_test_list[i]))
        f.write("\n")
    f.close()


if __name__ == "__main__":
#    generate_processed_data()
#    generate_train_test_data()
    train()
    print("test")