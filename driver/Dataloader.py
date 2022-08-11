from lib2to3.pgen2 import token
from basic.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable

def read_corpus(file_path, vocab=None):
    data = []
    with open(file_path, 'r', encoding="utf8") as infile:
        for sentence in readDepTree(infile, vocab):
            data.append(sentence)
    return data

def sentences_numberize(sentences, vocab, ignoreTree):
    for sentence in sentences:
        if ignoreTree:
            yield sentence2id_ignore_tree(sentence, vocab)
        else:
            yield sentence2id(sentence, vocab)

def sentence2id(sentence, vocab):
    result = []
    for dep in sentence:
        wordid = 0
        extwordid = 0
        #wordid = vocab.word2id(dep.form)
        #extwordid = vocab.extword2id(dep.form)
        tagid = vocab.tag2id(dep.tag)
        head = dep.head
        relid = vocab.rel2id(dep.rel)
        score = dep.score
        result.append([wordid, extwordid, tagid, head, relid, score])

    return result

def sentence2id_ignore_tree(sentence, vocab):
    result = []
    for dep in sentence:
        wordid = 0
        extwordid = 0
        #wordid = vocab.word2id(dep.form)
        #extwordid = vocab.extword2id(dep.form)
        tagid = vocab.tag2id(dep.tag)
        # head = dep.head
        # relid = vocab.rel2id(dep.rel)
        head = -1
        relid = -1
        score = dep.score
        result.append([wordid, extwordid, tagid, head, relid, score])

    return result



def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable(batch, vocab, ignoreTree=False):
    length = len(batch[0])
    batch_size = len(batch)
    for b in range(1, batch_size):
        if len(batch[b]) > length: length = len(batch[b])

    #words = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    #extwords = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    tags = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    masks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
    scores = Variable(torch.FloatTensor(batch_size, length).zero_(), requires_grad=False)
    heads = []
    rels = []
    lengths = []
    # scores = []

    b = 0
    for sentence in sentences_numberize(batch, vocab, ignoreTree):
        index = 0
        length = len(sentence)
        lengths.append(length)
        head = np.zeros((length), dtype=np.int32)
        rel = np.zeros((length), dtype=np.int32)
        # score = np.zeros((length), dtype=np.float32)
        for dep in sentence:
            #words[b, index] = dep[0]
            #extwords[b, index] = dep[1]
            if dep[2] == None:
                dep[2] = 0
            tags[b, index] = dep[2]
            head[index] = dep[3]
            rel[index] = dep[4]
            scores[b, index] = dep[5]
            masks[b, index] = 1
            index += 1
        b += 1
        heads.append(head)
        rels.append(rel)
        # scores.append(score)

    return tags, heads, rels, lengths, masks, scores

def batch_variable_depTree(trees, heads, rels, lengths, vocab):
    for tree, head, rel, length in zip(trees, heads, rels, lengths):
        sentence = []
        for idx in range(length):
            sentence.append(Dependency(idx, tree[idx].org_form, tree[idx].tag, head[idx], vocab.id2rel(rel[idx])))
        yield sentence

def token2ids(instances, tokenizer):
    for dep in instances:
        for w_info in dep:
            w_info.token_ids = tokenizer.encode(w_info.form, add_special_tokens=False)
            w_info.dens = [1 / len(w_info.token_ids) for idx in range(len(w_info.token_ids))] 

def token_variable(onebatch):
    b = len(onebatch)

    token_ids_list = []
    for dep in onebatch:
        token_ids = []
        for w_info in dep:
            token_ids += w_info.token_ids
        token_ids_list.append(token_ids)
    t = max([len(token_ids) for token_ids in token_ids_list])

    input_ids = np.zeros([b, t], dtype=np.long) 
    token_type_ids = np.zeros([b, t], dtype=np.long) 
    attention_mask = np.zeros([b, t], dtype=np.long) 

    for idx in range(b):
        token_len = len(token_ids_list[idx])
        for idy in range(token_len):
            input_ids[idx, idy] = token_ids_list[idx][idy]
            token_type_ids[idx, idy] = 0
            attention_mask[idx, idy] = 1

    input_ids = torch.tensor(input_ids)
    token_type_ids = torch.tensor(token_type_ids)
    attention_mask = torch.tensor(attention_mask)

    w = max([len(dep) for dep in onebatch])
    ts = max([len(w_info.token_ids) for dep in onebatch for w_info in dep])

    token_indexs = np.zeros([b, w, ts], dtype=np.long) 
    batch_dens = np.zeros([b, w, ts], dtype=np.float32) 

    for idx, dep in enumerate(onebatch):
        offset = 0
        for idy, w_info in enumerate(dep):
            for idz, den in enumerate(w_info.dens):
                token_indexs[idx, idy, idz] = offset 
                batch_dens[idx, idy, idz] = den
                offset += 1
    token_indexs =  torch.tensor(token_indexs).type(torch.int64)
    batch_dens =  torch.tensor(batch_dens)

    return (input_ids, token_type_ids, attention_mask), token_indexs, batch_dens