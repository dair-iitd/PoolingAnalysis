import ipdb
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append("my_models/")
from models import RNN
import custom_lstm
from grad_cam import *
from my_dataloader import *
from my_utils import *
from torchtext import data
from torchtext import datasets

import random
import time
import argparse
import copy
import math
import params

def myprint(s):
    print(s)
    return

parser = params.parse_args()
args = parser.parse_args()
args = add_config(args) if args.config_file != None else args
assert(args.mode == "test")
assert(args.task not in ["MNIST", "PMNIST"])

set_all_seeds_to(args.seed)

MAX_VOCAB_SIZE = 25_000 if(args.cap_vocab) else 1_00_000
print (MAX_VOCAB_SIZE)

device = torch.device('cuda:{0}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
if args.pool == 'last1' or args.pool == 'max1' or args.pool == 'mean1':
    custom_lstm.forget_bias = args.forget_bias


criterion = nn.CrossEntropyLoss()
accuracy = categorical_accuracy

import copy
# ipdb.set_trace()
args_load = copy.deepcopy(args)
args_load.lr = 2e-3
args_load.batch_size = 32
args.model_path = get_model_path(args_load)
if args.seed != 1234:
    model_dir = f"../models_{str(args.seed)}/" + args.task + '/' + args.pool + '/' + args.model_path
else:
    model_dir = "../models/" + args.task + '/' + args.pool + '/' + args.model_path

print(model_dir)

model_name = model_dir + '/best.pt'
if args.mode == "resume":
    print("Resume")
    model_name = model_dir + '/best_resume.pt'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# logf, train_acc_f, valid_acc_f, test_acc_f, aggregate_gradients_f, activations_f, gates_f, gates_gradients_f = get_all_logs(args,model_dir)

TEXT, LABEL, train_iterator, valid_iterator, test_iterator = get_data(args, MAX_VOCAB_SIZE, device)

vocab_size = len(TEXT.vocab) 
pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
output_dim = len(LABEL.vocab)

model = RNN(vocab_size = vocab_size,
            embedding_dim = args.embed_dim, 
            hidden_dim = args.hidden_dim, 
            output_dim = output_dim, 
            bidirectional = args.bidirectional, 
            pad_idx = pad_idx, 
            gpu_id = args.gpu_id, 
            pool = args.pool, 
            percent = None, 
            pos_vec = "none", 
            pos_wiki= "none", 
            dc = args.drop_connect)


if args.glove and args.use_embedding:
    pretrained_embeddings = TEXT.vocab.vectors
    myprint(pretrained_embeddings.shape)
    model.embedding.weight.data.copy_(pretrained_embeddings)
    # UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    # model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    # model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
if args.freeze_embedding:
    model.embedding.weight.requires_grad = False
    
model = model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

myprint(f'The model has {count_parameters(model):,} trainable parameters')

myprint('Data Loading done!')

criterion = criterion.to(device)


def vector_to_text(text):
    #text = [vectors, batch size]
    l = ['']*text.shape[1]
    for i in range(text.shape[1]):
        for j in range(text.shape[0]):
            l[i] = l[i] + TEXT.vocab.itos[text[j][i]] + " "
    return l


def evaluate(model, iterator, criterion, return_attention_weights = False):
    # ipdb.set_trace()
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    l_sum = 0
    m_sum = 0
    r_sum = 0
    with torch.no_grad():
        for i,batch in enumerate(iterator):
            text, text_lengths = batch.text
            output = model(text, text_lengths, use_embedding = args.use_embedding, return_attention_weights = return_attention_weights)
            predictions = output[0].squeeze(1)
            if return_attention_weights:
                ipdb.set_trace()
                att = output[-1]
                lgh = text_lengths[0] 
                l_sum += att[:lgh//3].sum()
                m_sum += att[lgh//3:2*lgh//3].sum()
                r_sum += att[2*lgh//3:].sum()
            loss = criterion(predictions, batch.label)
            acc = accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    print(l_sum/ len(iterator),m_sum/ len(iterator),r_sum/ len(iterator))
    return epoch_loss / len(iterator), epoch_acc /len(iterator)

def text_to_vector(text):
    #text = ""
    # ipdb.set_trace()
    l = text.split(" ")
    vec = torch.zeros(len(l))
    for i in range(len(l)):
        vec[i] = TEXT.vocab.stoi[l[i]] 
    return vec

import json
wiki_simple = '../data/wiki/simple_wiki_sort.json'
with open(wiki_simple) as json_file:
    y_test = json.load(json_file)
keys_test = list(y_test.keys())
num_keys_test = len(keys_test)

def get_str(lgh):  
    curr = 0
    s = ''
    k = keys_test
    nk = num_keys_test
    y_ = y_test
        
    while (curr < lgh):
        index = np.random.randint(0,nk)
        size = k[index]
        d_list = y_[size]
        s1 = random.choice(d_list)
        curr = curr + int(size) 
        s = s + " " + s1
        if curr > lgh:
            break
    return s


def get_wiki_vec(lgh):
    # ipdb.set_trace()
    s1 = get_str(lgh)
    s2 = get_str(lgh)
    v1 = text_to_vector(s1).long()
    v2 = text_to_vector(s2).long()
    return v1, v2

def wiki_appended_text(text, text_lengths, percent, pos_wiki):
    # ipdb.set_trace()
    lgh = (2*text_lengths[0]*percent//100).item()
    text = text.t()
    new_text = torch.zeros(text.shape[0], 2*lgh+text.shape[1], dtype = torch.long).to(device)
    v1, v2 = get_wiki_vec(lgh)

    for i in range(text.shape[0]):
        real_len = text_lengths[i] #RL
        v1, v2 = v1[:lgh].to(device), v2[:lgh].to(device)
        if pos_wiki == "left": #left must be preserved (X)
            new_text[i] = torch.cat((text[i][:real_len], v1, v2, text[i][real_len:]))
        elif pos_wiki == "mid":
            new_text[i] = torch.cat((v1, text[i][:real_len], v2, text[i][real_len:]))
        elif pos_wiki == "right":
            new_text[i] = torch.cat((v1, v2, text[i][:real_len], text[i][real_len:]))
    return new_text.t()

            

def evaluate_wiki_attack(model, iterator, criterion, percent, pos_wiki, return_attention_weights = False):
    # ipdb.set_trace()
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    l_sum,m_sum,r_sum = 0,0,0
    with torch.no_grad():
        for i,batch in enumerate(iterator):
            text, text_lengths = batch.text
            text = wiki_appended_text(text, text_lengths, percent, pos_wiki)
            text_lengths = text_lengths + text_lengths[0]*4*percent//100
            output = model(text, text_lengths, use_embedding = args.use_embedding, return_attention_weights = return_attention_weights)
            predictions = output[0].squeeze(1)
            if return_attention_weights:

                original_length = batch.text[1]
                text_start_pos = original_length[0]*2*percent//100
                text_end_pos = original_length[0] + original_length[0]*2*percent//100
                att = output[-1]
                att = att.abs()
 
                l_sum += att[:text_start_pos].sum().item()
                m_sum += att[text_start_pos:text_end_pos].sum().item()
                r_sum += att[text_end_pos:].sum().item()
            loss = criterion(predictions, batch.label)
            acc = accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    print(l_sum/ len(iterator),m_sum/ len(iterator),r_sum/ len(iterator))
    return epoch_loss / len(iterator), epoch_acc /len(iterator)
 

def evaluate_NWI(model,iterator, criterion, k, len_range, req_ex = 500): 
    epoch_loss = 0
    epoch_acc = 0
    n = 0
    k = k
    model.eval()
    req_batches = req_ex//args.batch_size
    losses_final = np.zeros((req_batches,100))
    with torch.no_grad():
        len_range = len_range
        for batch in iterator:
            text, text_lengths = batch.text
            if text_lengths[0] < len_range: 
                continue
            # print(text_lengths[0])
            batch_size = text.shape[1]
            losses_stored = torch.zeros(len_range//k)
            losses_stored_max = torch.zeros(batch_size, len_range//k)
            
            
            out, sent_emb, word_emb = model(text, text_lengths)
            orig_loss = criterion(out.squeeze(1), batch.label)
            
            # ipdb.set_trace()
            text_temp = text.clone()
            num_steps = text_lengths[0]//k
            text_temp = text_temp[:num_steps*k]
            text_temp_repeated = text_temp.unsqueeze(0).repeat(num_steps,1,1)
            # [num_steps,sent_len,batch]
            mask = torch.ones_like(text_temp_repeated)
            pos=0
            for i in range(0, num_steps*k, k):
                mask[pos,i:i+k,:] = 0
                pos+=1


            text_temp_repeated = mask*text_temp_repeated
            # [sent_len,num_steps,batch]
            text_temp_repeated = text_temp_repeated.permute(1,0,2)
            # [sent_len,num_steps*batch] ~ [batch,batch,batch...]
            text_temp_repeated = text_temp_repeated.reshape(text_temp_repeated.shape[0],-1)

            out, _, _ = model(text_temp_repeated, 0)
            # [num_steps*batch]  ~ [batch,batch,batch...]
            labels = batch.label.repeat(num_steps)
            loss = criterion(out.squeeze(1), labels)
            orig_loss_repeated = orig_loss.repeat(num_steps)
            losses_stored_max = (abs(loss-orig_loss_repeated)).reshape(num_steps,batch_size).permute(1,0)
            max_losses = losses_stored_max.max(dim = 1)[0].unsqueeze(1)
            max_losses[max_losses==0] = 1
            losses_stored = (losses_stored_max/max_losses).mean(dim = 0)
            fp = losses_stored.tolist()
            x = list(range(100))
            xp = np.linspace(0,99,len(fp))
            interp_grads = np.interp(x, xp, fp)
            losses_final[n] = interp_grads
            n+=1
            torch.cuda.empty_cache()
            if n == req_batches:
                break
        return losses_final.mean(axis = 0), n

 

if args.vec == 2:
    checkpoint = torch.load(model_name, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict']) 
    import time
    start = time.time()
    # test_loss, test_acc = evaluate(model, test_iterator, criterion)
    # print ("Time taken = ", time.time() - start)
    # myprint(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    print("Test with wiki data on normal trained models")
    logwiki = open(model_dir+'/log_wiki_on_noneX3' + '.txt','w')
    print(model_name)
    for pos_wiki in ['mid', 'left', 'right']:
        for percent in [100]:#,5,10,15,20,30,40,50,60,70,80,90,100]:
            start = time.time()
            test_loss, test_acc = evaluate_wiki_attack(model, test_iterator, criterion, percent, pos_wiki, return_attention_weights = True)
            print ("Time taken = ", time.time() - start)
            print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
            logwiki.write(f'{test_acc*100:.2f}%\n')   

elif args.vec == 3:
    checkpoint = torch.load(model_name, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict']) 
    import time
    start = time.time()
    # test_loss, test_acc = evaluate(model, test_iterator, criterion)
    # print ("Time taken = ", time.time() - start)
    # myprint(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    print("Test with wiki data on normal trained models")
    logwiki = open(model_dir+'/log_wiki_on_noneX3' + '.txt','w')
    print(model_name)
    for pos_wiki in ['mid', 'left', 'right']:
        for percent in [1]:#,5,10,15,20,30,40,50,60,70,80,90,100]:
            start = time.time()
            test_loss, test_acc = evaluate_wiki_attack(model, test_iterator, criterion, percent, pos_wiki)#, return_attention_weights = True)
            print ("Time taken = ", time.time() - start)
            print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
            logwiki.write(f'{test_acc*100:.2f}%\n')   
           
elif args.NWI:
    import time
    def get_NWI(r1):
        r1 = r1 - min(r1)
        r1 /= max(r1)
        return list(r1)

    criterion = nn.CrossEntropyLoss(reduction = 'none')
    kl_pair = [(5,400)]#,(10,800)]
    if args.req_ex == 400:
        kl_pair = [(3,50)]
    for (k,len_range) in kl_pair:
        start_time = time.time()
        l_nwi = np.zeros(100)
        logkl = open(model_dir+'/log_k_'+str(k) + '_l_' + str(len_range) + '.txt','w')
        logkl_originial = open(model_dir+'/log_k_'+str(k) + '_l_' + str(len_range) + '_original.txt','w')
        seeds = [101,119,7,37,1234]
        for seed in seeds:
            if seed != 1234:
                model_dir = f"../models_{str(seed)}/" + args.task + '/' + args.pool + '/' + args.model_path
            else:
                model_dir = "../models/" + args.task + '/' + args.pool + '/' + args.model_path
            model_name = model_dir + '/best.pt'
            print(model_name)
            checkpoint = torch.load(model_name, map_location = device)
            model.load_state_dict(checkpoint['model_state_dict'])  
            l,n = evaluate_NWI(model, valid_iterator, criterion, k, len_range, req_ex = args.req_ex)
            l_nwi += np.array(l)

        l_nwi = l_nwi / len(seeds)

        for element in list(l_nwi):
            logkl_originial.write(str(element) + '\n')
        logkl_originial.close()

        l_nwi = get_NWI(l_nwi)
        for element in l_nwi:
            logkl.write(str(element) + '\n')
        logkl.close()

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        myprint(f'KL Time: {epoch_mins}m {epoch_secs}s')

elif args.ood:
    checkpoint = torch.load(model_name, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict']) 
    import time
    start = time.time()
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print ("Time taken = ", time.time() - start)
    myprint(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    print("Test with wiki data on normal trained models")
    print(model_name)
    for pos_wiki in ['mid', 'left', 'right']:
        args.wiki  = pos_wiki
        test_iterator = get_ood_test_data(args, device, TEXT, LABEL)
        test_loss, test_acc = evaluate(model, test_iterator, criterion, return_attention_weights = True)
        print ("Time taken = ", time.time() - start)
        myprint(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

else:
    checkpoint = torch.load(model_name, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict']) 
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    myprint(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    test_acc_f.write(str(test_acc*100)+'\n')
    test_acc_f.flush()
    test_acc_f.close()
    

