from enum import auto
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from basic.Config import *
from modules.Model import *
from driver.Parser import *
from driver.Dataloader import *
from basic.Optimizer import *
import pickle

from modules.BertTune import *
from transformers.models.auto.tokenization_auto import AutoTokenizer

def train(data, dev_data, test_data, parser, vocab, config):
    auto_param = list(parser.plm_extractor.parameters())
    parser_param = list(parser.model.parameters())

    model_param = [{'params': auto_param, 'lr': config.plm_learning_rate},
                {'params': parser_param, 'lr': config.learning_rate}]

    optimizer = Optimizer(model_param, config)

    global_step = 0
    best_UAS = 0
    best_LAS = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_arc_correct, overall_label_correct, overall_total_arcs = 0, 0, 0
        for onebatch in data_iter(data, config.train_batch_size, True):
            inputs, token_indexs, dens = token_variable(onebatch)
            tags, heads, rels, lengths, masks, scores = \
                batch_data_variable(onebatch, vocab)
            parser.train()

            parser.forward(
                inputs, token_indexs, dens,
                tags, masks
            )
            loss = parser.compute_loss(heads, rels, lengths, scores, config.threshold)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            arc_correct, label_correct, total_arcs = parser.compute_accuracy(heads, rels)
            overall_arc_correct += arc_correct
            overall_label_correct += label_correct
            overall_total_arcs += total_arcs
            uas = overall_arc_correct.item() * 100.0 / overall_total_arcs
            las = overall_label_correct.item() * 100.0 / overall_total_arcs
            during_time = float(time.time() - start_time)
            print("Step:%d, ARC:%.2f, REL:%.2f, Iter:%d, batch:%d, length:%d,time:%.2f, loss:%.2f" \
                %(global_step, uas, las, iter, batch_iter, overall_total_arcs, during_time, loss_value))

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(auto_param + parser_param, max_norm=config.clip)

                optimizer.step()
                parser.model.zero_grad()       
                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                arc_correct, rel_correct, arc_total, dev_uas, dev_las = \
                    evaluate(dev_data, parser, vocab, config.dev_file + '.' + str(global_step))
                print("Dev: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                      (arc_correct, arc_total, dev_uas, rel_correct, arc_total, dev_las))
                arc_correct, rel_correct, arc_total, test_uas, test_las = \
                    evaluate(test_data, parser, vocab, config.test_file + '.' + str(global_step))
                print("Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                      (arc_correct, arc_total, test_uas, rel_correct, arc_total, test_las))
                if dev_las > best_LAS:
                    print("Exceed best las: history = %.2f, current = %.2f" %(best_LAS, dev_las))
                    best_LAS = dev_las
                    if config.save_after > 0 and iter > config.save_after:
                        torch.save(parser.model.state_dict(), config.save_model_path)


def evaluate(data, parser, vocab, outputFile):
    start = time.time()
    parser.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        inputs, token_indexs, dens = token_variable(onebatch)
        tags, heads, rels, lengths, masks, scores = batch_data_variable(onebatch, vocab, ignoreTree=True)
        count = 0
        arcs_batch, rels_batch = parser.parse(
            inputs, token_indexs, dens,
            tags, lengths, masks
        )
        for tree in batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, vocab):
            printDepTree(output, tree)
            # arc_total, arc_correct, rel_total, rel_correct = evalDepTree(tree, onebatch[count])
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(onebatch[count], tree)
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct
            count += 1

    output.close()

    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test


    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))

    return arc_correct_test, rel_correct_test, arc_total_test, uas, las


if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../ctb.parser.cfg.debug')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)


    print("Loading plm: ", config.plm_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)
    plm_extractor = BertModelExtractor(config.plm_dir, config, tokenizer)
    print("Loading ok")

    # vocab = creatVocab(config.train_file, config.min_occur_count)
    vocab = creatVocab([config.train_file, config.train_target_file]
                       , config.min_occur_count)
    #vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # print(config.use_cuda)
    config.plm_hidden_size = plm_extractor.auto_model.base_model.config.hidden_size
    model = ParserModel(vocab, config)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        plm_extractor = plm_extractor.cuda()
        model = model.cuda()

    parser = BiaffineParser(plm_extractor, model, vocab.ROOT)

    data = read_corpus(config.train_file, vocab)
    data_target = read_corpus(config.train_target_file, vocab)
    data.extend(data_target)
    dev_data = read_corpus(config.dev_file, vocab)
    test_data = read_corpus(config.test_file, vocab)

    token2ids(data, tokenizer)
    token2ids(dev_data, tokenizer)
    token2ids(test_data, tokenizer)

    train(data, dev_data, test_data, parser, vocab, config)
