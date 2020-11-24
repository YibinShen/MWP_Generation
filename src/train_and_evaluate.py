from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import time

MAX_OUTPUT_LENGTH1 = 120
MAX_OUTPUT_LENGTH2 = 45
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output1, all_output2):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output1 = all_output1
        self.all_output2 = all_output2


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_tree_input(target, decoder_output, num_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = num_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, num_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = num_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    batch_size = encoder_outputs.size(1)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index.bool(), 0.0), masked_index


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


def train_problem(keywords_batch, keywords_length, problem_batch, problem_length, equation_batch, equation_length, 
                  prototype_keywords_batch, prototype_keywords_length, prototype_problem_batch, prototype_problem_length, 
                  prototype_graph1_batch, prototype_graph2_batch, 
                  num_size_batch, num_pos_batch, num_stack_batch, generate_num_ids, 
                  keywords_encoder, prototype_encoder, decoder, predict, generate, merge, 
                  keywords_encoder_optimizer, prototype_encoder_optimizer, decoder_optimizer, 
                  predict_optimizer, generate_optimizer, merge_optimizer, 
                  problem_lang, equation_lang, beam_size=5, use_teacher_forcing=1, lenpen1=1):
    # sequence mask for attention
    prototype_problem_seq_mask = []
    prototype_problem_max_len = max(prototype_problem_length)
    for i in prototype_problem_length:
        prototype_problem_seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, prototype_problem_max_len)])
    prototype_problem_seq_mask = torch.ByteTensor(prototype_problem_seq_mask)
    
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    keywords_var = torch.LongTensor(keywords_batch).transpose(0, 1)
    target1_var = torch.LongTensor(problem_batch).transpose(0, 1)
    target2_var = torch.LongTensor(equation_batch).transpose(0, 1)
    prototype_keywords_var = torch.LongTensor(prototype_keywords_batch).transpose(0, 1)
    prototype_problem_var = torch.LongTensor(prototype_problem_batch).transpose(0, 1)
    prototype_graph1_var = torch.LongTensor(prototype_graph1_batch)
    prototype_graph2_var = torch.LongTensor(prototype_graph2_batch)
#    prototype_index = torch.LongTensor(prototype_belong_index)

    keywords_encoder.train()
    prototype_encoder.train()
    decoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        keywords_var = keywords_var.cuda()
        target1_var = target1_var.cuda()
        target2_var = target2_var.cuda()
        prototype_keywords_var = prototype_keywords_var.cuda()
        prototype_problem_var = prototype_problem_var.cuda()
        prototype_problem_seq_mask = prototype_problem_seq_mask.cuda()
        prototype_graph1_var = prototype_graph1_var.cuda()
        prototype_graph2_var = prototype_graph2_var.cuda()
#        prototype_index = prototype_index.cuda()

    # Zero gradients of both optimizers
    keywords_encoder_optimizer.zero_grad()
    prototype_encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    
    keywords_outputs, keywords_hidden = keywords_encoder(keywords_var, keywords_length)
    prototype_outputs = prototype_encoder(prototype_keywords_var, prototype_keywords_length,
                                          prototype_problem_var, prototype_problem_length, 
                                          prototype_graph1_var, prototype_graph2_var, 
                                          keywords_outputs[-1], prototype_problem_seq_mask)
    
    decoder_hidden = keywords_hidden[::2].contiguous()  # Use last (forward) hidden state from encoder
    loss_0, encoder_outputs, _ = train_attn_problem(keywords_outputs, decoder_hidden, prototype_outputs, target1_var, problem_length, 
                                                    decoder, problem_lang, beam_size, use_teacher_forcing, lenpen1)
    problem_output = encoder_outputs[-1]
    
    problem_seq_mask = []
    max_len = max(problem_length)-1
    for i in problem_length:
        problem_seq_mask.append([0 for _ in range(i-1)] + [1 for _ in range(i-1, max_len)])
    problem_seq_mask = torch.ByteTensor(problem_seq_mask)
    
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_num_ids)
    for i in num_size_batch:
        d = i + len(generate_num_ids)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)
    
    num_pos_pad = []
    max_num_pos_size = max(num_size_batch)
    for i in range(len(num_pos_batch)):
        temp = num_pos_batch[i] + [-1] * (max_num_pos_size-len(num_pos_batch[i]))
        num_pos_pad.append(temp)
    num_pos_pad = torch.LongTensor(num_pos_pad)
    
    if USE_CUDA:
        problem_seq_mask = problem_seq_mask.cuda()
        num_mask = num_mask.cuda()
        num_pos_pad = num_pos_pad.cuda()
#    
    num_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, num_pos_batch, 
                                                                       max(num_size_batch), decoder.hidden_size)
    num_outputs = num_encoder_outputs.masked_fill_(masked_index.bool(), 0.0)
#    
    loss_1 = train_tree_equation(encoder_outputs, problem_output, num_outputs, 
                                 target2_var, equation_length, problem_seq_mask, num_mask, copy.deepcopy(num_stack_batch), 
                                 equation_lang, predict, generate, merge)
    loss = loss_0 + loss_1
    loss.backward()
    
    keywords_encoder_optimizer.step()
    prototype_encoder_optimizer.step()
    decoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    
    return loss.item()  # , loss_0.item(), loss_1.item()


def train_attn_problem(keywords_outputs, decoder_hidden, prototype_outputs, target_var, target_length,
                       decoder, problem_lang, beam_size, use_teacher_forcing, lenpen):
    batch_size = keywords_outputs.size(1)
    decoder_input = torch.LongTensor([problem_lang.word2index["SOS"]] * batch_size)

    max_target_length = max(target_length)
    all_decoder_outputs1 = torch.zeros(max_target_length, batch_size, decoder.hidden_size)
    all_decoder_outputs2 = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs1 = all_decoder_outputs1.cuda()
        all_decoder_outputs2 = all_decoder_outputs2.cuda()
    
    teacher_flag = 1
    if random.random() < use_teacher_forcing:
        teacher_flag = 1
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoder_output1, decoder_output2, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, keywords_outputs, prototype_outputs)
            all_decoder_outputs1[t] = decoder_output1
            all_decoder_outputs2[t] = decoder_output2
            decoder_input = target_var[t]
    else:
        teacher_flag = 0
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs1, all_decoder_outputs2))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs1 = torch.zeros(max_target_length, batch_size * beam_len, decoder.hidden_size)
            all_outputs2 = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs1 = all_outputs1.cuda()
                all_outputs2 = all_outputs2.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                if USE_CUDA:
                    decoder_input = decoder_input.cuda()

                decoder_output1, decoder_output2, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, keywords_outputs, prototype_outputs)

                lp = (5+t+1) ** lenpen / (5+1) ** lenpen
                score = f.log_softmax(decoder_output2, dim=1) / lp
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output1[t] = decoder_output1
                beam_list[b_idx].all_output2[t] = decoder_output2
                all_outputs1[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output1
                all_outputs2[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output2
            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output1 = all_outputs1.index_select(1, indices)
                temp_output2 = all_outputs2.index_select(1, indices)
                
                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output1, temp_output2))
            
        all_decoder_outputs1 = beam_list[0].all_output1
        all_decoder_outputs2 = beam_list[0].all_output2


    if USE_CUDA:
        target_var = target_var.cuda()
    
    loss = masked_cross_entropy_smooth(
        all_decoder_outputs2.transpose(0, 1).contiguous(),  # -> batch x seq
        target_var.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    )
#    return beam_list[0]
    return loss, all_decoder_outputs1[:-1], all_decoder_outputs2


def train_tree_equation(encoder_outputs, problem_output, num_outputs, equation_var, equation_length, 
                        seq_mask, num_mask, num_stack_batch, 
                        equation_lang, predict, generate, merge):
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    batch_size = encoder_outputs.size(1)
    max_target_length = max(equation_length)
    all_node_outputs = []
    num_start = equation_lang.num_start
    unk = equation_lang.word2index["UNK"]
    
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    padding_hidden = torch.zeros((1, predict.hidden_size))
    
    if USE_CUDA:
        padding_hidden = padding_hidden.cuda()
    
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, num_outputs, padding_hidden, seq_mask, num_mask)

        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(equation_var[t].tolist(), outputs, 
                                                       num_stack_batch, num_start, unk)
        equation_var[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, equation_var[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    equation_var = equation_var.transpose(0, 1).contiguous()
    if USE_CUDA:
        all_node_outputs = all_node_outputs.cuda()
        equation_var = equation_var.cuda()

    loss = masked_cross_entropy(all_node_outputs, equation_var, equation_length)
    
    return loss


def evaluate_problem(keywords_batch, keywords_length, 
                     prototype_keywords_batch, prototype_keywords_length, 
                     prototype_problem_batch, prototype_problem_length, 
                     prototype_graph1_batch, prototype_graph2_batch, 
                     generate_num_ids, copy_nums, 
                     keywords_encoder, prototype_encoder, 
                     decoder, predict, generate, merge, 
                     problem_lang, equation_lang, beam_size=5, 
                     max_length1=MAX_OUTPUT_LENGTH1, lenpen1=1, max_length2=MAX_OUTPUT_LENGTH2):

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    keywords_var = torch.LongTensor(keywords_batch).transpose(0, 1)
    prototype_keywords_var = torch.LongTensor(prototype_keywords_batch).transpose(0, 1)
    prototype_problem_var = torch.LongTensor(prototype_problem_batch).transpose(0, 1)
    prototype_problem_seq_mask = torch.ByteTensor(1, prototype_problem_length[0]).fill_(0)
    prototype_graph1_var = torch.LongTensor(prototype_graph1_batch)
    prototype_graph2_var = torch.LongTensor(prototype_graph2_batch)
#    prototype_index = torch.LongTensor(prototype_belong_index)
    
    # Set to not-training mode to disable dropout
    keywords_encoder.eval()
    prototype_encoder.eval()
    decoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    if USE_CUDA:
        keywords_var = keywords_var.cuda()
        prototype_keywords_var = prototype_keywords_var.cuda()
        prototype_problem_var = prototype_problem_var.cuda()
        prototype_problem_seq_mask = prototype_problem_seq_mask.cuda()
        prototype_graph1_var = prototype_graph1_var.cuda()
        prototype_graph2_var = prototype_graph2_var.cuda()
#        prototype_index = prototype_index.cuda()
    
    keywords_outputs, keywords_hidden = keywords_encoder(keywords_var, keywords_length)
    prototype_outputs = prototype_encoder(prototype_keywords_var, prototype_keywords_length, 
                                          prototype_problem_var, prototype_problem_length, 
                                          prototype_graph1_var, prototype_graph2_var, 
                                          keywords_outputs[-1], prototype_problem_seq_mask)
    decoder_hidden = keywords_hidden[::2].contiguous()  # Use last (forward) hidden state from encoder
    
    problem_beam = evaluate_attn_problem(keywords_outputs, decoder_hidden, prototype_outputs,
                                         decoder, problem_lang, beam_size, max_length1, lenpen1)
    problem_length = len(problem_beam.all_output2)-1
    
    if problem_length:
        problem_seq_mask = torch.ByteTensor(1, problem_length).fill_(0)
        encoder_outputs = problem_beam.all_output1[:problem_length].unsqueeze(1)
        problem_output = encoder_outputs[-1]
        encoder_problem = problem_beam.all_output2[:-1]
        num_pos_batch = np.where(np.array(encoder_problem)==problem_lang.word2index["NUM"])[0].tolist()
        num_mask = torch.ByteTensor(1, len(num_pos_batch) + len(generate_num_ids)).fill_(0)
        
        if USE_CUDA:
            encoder_outputs = encoder_outputs.cuda()
            problem_output = problem_output.cuda()
            problem_seq_mask = problem_seq_mask.cuda()
            num_mask = num_mask.cuda()
        
        num_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, [num_pos_batch], 
                                                                           len(num_pos_batch), decoder.hidden_size)
        num_outputs = num_encoder_outputs.masked_fill_(masked_index.bool(), 0.0)
        
        equation_beam = evaluate_tree_equation(encoder_outputs, problem_output, num_outputs, problem_seq_mask, num_mask, 
                                               equation_lang, predict, generate, merge, beam_size, max_length2)
        return problem_beam, equation_beam
    else:
        return problem_beam, 0


def evaluate_attn_problem(keywords_outputs, decoder_hidden, prototype_outputs, 
                          decoder, problem_lang, beam_size, max_length, lenpen):
    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([problem_lang.word2index["SOS"]])  # SOS
    beam_list = list()
    score = 0
    all_decoder_outputs1 = torch.zeros(1, max_length, decoder.hidden_size)
    beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs1, []))

    # Run through decoder
    for di in range(max_length):
        temp_list = list()
        beam_len = len(beam_list)
        for xb in beam_list:
            if int(xb.input_var[0]) == problem_lang.word2index["EOS"]:
                temp_list.append(xb)
                beam_len -= 1
        if beam_len == 0:
            return beam_list[0]
        beam_scores = torch.zeros(decoder.output_size * beam_len)
        hidden_size_0 = decoder_hidden.size(0)
        hidden_size_2 = decoder_hidden.size(2)
        all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
        if USE_CUDA:
            beam_scores = beam_scores.cuda()
            all_hidden = all_hidden.cuda()
        all_outputs1 = torch.zeros(beam_len, max_length, decoder.hidden_size)
        all_outputs2 = []
        current_idx = -1

        for b_idx in range(len(beam_list)):
            decoder_input = beam_list[b_idx].input_var
            if int(decoder_input[0]) == problem_lang.word2index["EOS"]:
                continue
            current_idx += 1
            decoder_hidden = beam_list[b_idx].hidden

            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            
            decoder_output1, decoder_output2, decoder_hidden = decoder(
                decoder_input, decoder_hidden, keywords_outputs, prototype_outputs)
            # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
            lp = (5+di+1) ** lenpen / (5+1) ** lenpen
            score = f.log_softmax(decoder_output2, dim=1) / lp
            score += beam_list[b_idx].score
            beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
            all_hidden[current_idx] = decoder_hidden
            
            beam_list[b_idx].all_output1[di] = decoder_output1
            all_outputs1[current_idx] = beam_list[b_idx].all_output1
            all_outputs2.append(beam_list[b_idx].all_output2)
        topv, topi = beam_scores.topk(beam_size)

        for k in range(beam_size):
            word_n = int(topi[k])
            word_input = word_n % decoder.output_size
            temp_input = torch.LongTensor([word_input])
            indices = int(word_n / decoder.output_size)

            temp_hidden = all_hidden[indices]
            temp_output1 = all_outputs1[indices]
            temp_output2 = all_outputs2[indices]+[word_input]
            temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output1, temp_output2))

        temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

        if len(temp_list) < beam_size:
            beam_list = temp_list
        else:
            beam_list = temp_list[:beam_size]
    return beam_list[0]


def evaluate_tree_equation(encoder_outputs, problem_output, num_outputs, seq_mask, num_mask, 
                           equation_lang, predict, generate, merge, beam_size, max_length):
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    batch_size = encoder_outputs.size(1)
    num_start = equation_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    padding_hidden = torch.zeros((1, predict.hidden_size))
    
    if USE_CUDA:
        padding_hidden = padding_hidden.cuda()
    
    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, num_outputs, padding_hidden,
                seq_mask, num_mask)

            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0]
