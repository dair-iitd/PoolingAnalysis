import torch
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.optim as optim
import ipdb
import dill as pickle
import numpy as np
import random



def get_wiki_test_data(args, device, TEXT, LABEL):
    #Vocabulary must stay consistent. fields can not be extended again
    task  = args.task
    test_path = {"left":"test_l2.json", "mid":"test_m2.json", "right":"test_r2.json"}
    test_path = test_path[args.wiki]
    
    try:
        # ipdb.set_trace()
        file = open("../data/"+task+f"/dump_{args.wiki}_2.pkl","rb")
        pickle_load = pickle.load(file)
        fields={'s': ('text', TEXT), 'class': ('label', LABEL)}
        fields, field_dict = [], fields
        for field in field_dict.values():
            if isinstance(field, list):
                fields.extend(field)
            else:
                fields.append(field) 

        test_data = data.Dataset(pickle_load, fields=fields)

    except:
        file = open("../data/"+task+f"/dump_{args.wiki}_2.pkl","wb")
        test_data = data.TabularDataset(
                path='../data/'+task + '/' + test_path, format='json',
                fields={'s': ('text', TEXT), 'class': ('label', LABEL)})



        pickle.dump(list(test_data), file)
    file.close()

    test_iterator = data.BucketIterator(
        dataset=test_data, batch_size=args.batch_size,
        sort_within_batch = True, sort_key = lambda x: len(x.text),
        device = device)

    return test_iterator


def get_ood_test_data(args, device, TEXT, LABEL):
    #Vocabulary must stay consistent. fields can not be extended again
    task  = args.task
    test_path = {"left":"test_l2.json", "mid":"test_m2.json", "right":"test_r2.json"}
    test_path = test_path[args.wiki]
    
        # ipdb.set_trace()
    file = open("../data/"+task+f"/dump_{args.wiki}.pkl","rb")
    pickle_load = pickle.load(file)
    # train_list, valid_list, test_list = pickle_load
    fields={'s': ('text', TEXT), 'class': ('label', LABEL)}
    fields, field_dict = [], fields
    for field in field_dict.values():
        if isinstance(field, list):
            fields.extend(field)
        else:
            fields.append(field) 

    train_list, valid_list, test_list = pickle_load

    test_data = data.Dataset(test_list, fields=fields)

    file.close()

    test_iterator = data.BucketIterator(
        dataset=test_data, batch_size=args.batch_size,
        sort_within_batch = True, sort_key = lambda x: len(x.text),
        device = device)

    return test_iterator



def load_pickle(args, TEXT, LABEL):
    file = open("../data/"+args.task+f"/dump_{args.wiki}.pkl","rb") if not args.debug else open("../data/"+args.task+"/debug_dump.pkl","rb")
    train_list, valid_list, test_list = pickle.load(file)
    file.close()
    fields={'s': ('text', TEXT), 'class': ('label', LABEL)}
    fields, field_dict = [], fields
    for field in field_dict.values():
        if isinstance(field, list):
            fields.extend(field)
        else:
            fields.append(field) 
    train_data, valid_data, test_data = data.Dataset(train_list, fields=fields), data.Dataset(valid_list, fields=fields), data.Dataset(test_list, fields=fields)
    return train_data, valid_data, test_data

def dump_pickle(args, TEXT, LABEL):
    train_path = {"none": "train.json", "left":"train_l.json", "mid":"train_m.json", "right":"train_r.json"}
    train_path = train_path[args.wiki]
    test_path = {"none": "test.json", "left":"test_l.json", "mid":"test_m.json", "right":"test_r.json"}
    test_path = test_path[args.wiki]
    dev_path = {"none": "dev.json", "left":"dev_l.json", "mid":"dev_m.json", "right":"dev_r.json"}
    dev_path = dev_path[args.wiki]
   
    task = args.task
    print ("WARNING: Pickle Load Unsuccessful. Training time will increase")
    if args.debug:
        train_path = dev_path = test_path = 'small.json'

    train_data, valid_data, test_data = data.TabularDataset.splits(
            path='../data/'+task, train=train_path,
            validation=dev_path, test=test_path, format='json',
            fields={'s': ('text', TEXT), 'class': ('label', LABEL)})

    file = open("../data/"+task+f"/dump_{args.wiki}.pkl","wb") if not args.debug else open("../data/"+task+"/debug_dump.pkl","wb")
    train_list, valid_list, test_list = list(train_data), list(valid_data), list(test_data)
    random.shuffle(train_list); random.shuffle(valid_list); random.shuffle(test_list)
    pickle.dump([train_list[:25000], valid_list[:25000], test_list[:25000]], file)
    file.close()



def get_data(args, MAX_VOCAB_SIZE, device):

    task  = args.task
    print(task)
    TEXT = data.Field(tokenize = 'spacy', include_lengths = True) 
    LABEL = data.LabelField()

    try:
        train_data, valid_data, test_data = load_pickle(args, TEXT, LABEL)
    except:
        dump_pickle(args, TEXT, LABEL)
        train_data, valid_data, test_data = load_pickle(args, TEXT, LABEL)
    num_examples = int(1000*float(args.data_size[:-1]))

    train_data.examples = train_data.examples[:num_examples]
    if not args.gradients:
        valid_data.examples = valid_data.examples[:num_examples]
    LABEL.build_vocab(train_data)
    
    vec = "glove.6B.100d" if args.glove else None
    unk_init = torch.Tensor.normal_ if args.glove else None

    TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = vec, unk_init = unk_init)    
    train_iterator, valid_iterator, test_iterator \
                                = data.BucketIterator.splits((train_data, valid_data, test_data),     
                                                batch_size = args.batch_size, 
                                                sort_within_batch = True, 
                                                sort_key = lambda x: len(x.text),
                                                device = device)

    return TEXT, LABEL, train_iterator, valid_iterator, test_iterator



