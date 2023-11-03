"""
Pretrain Stage1
There are two Functions implemented in this code:
1. pretrain a BERT in stage1
2. use this pretrained BERT to generate fake data for stage 2 replay
"""
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup

# Local imports
import base_models
from Dataset import MixedData
from utils import *


# If you have finished pretrain, you can load it
LOAD_CHECKPOINT = True

# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder paths
STORE_FOLDER = "1103-stage1"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
REPLAY_FILE = 'replay-data-12s-8*8.pth'
REPLAY_FILE_PATH = os.path.join(STORE_PATH, REPLAY_FILE)
CONFIG_PATH = 'config/bert.json'

# layers to generate fake data 
NUM_REPLAY_LAYERS = 12 # range from 1-12, this layer range start from the lowest (layer 0)
ONLY_LAST_LAYER = False # set true will disable NUM_LAYERS
NUM_CLUSTERS = 8
DATA_PER_CLUSTER = 8
SAMPLE_BATCHES = 20

# training parameters
num_epochs = 50
lr = 1e-4
weight_decay = 0


def validate(model, val_loader, accelerator):
    losses = []
    for i, batch in enumerate(val_loader):    
        with torch.no_grad():
            loss, _ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity


def main():
    
    config = BertConfig.from_json_file(CONFIG_PATH)

    dataset = MixedData(config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    
    model = base_models.BertForMLM(config)
    if LOAD_CHECKPOINT:
        checkpoint = torch.load(os.path.join(STORE_PATH, 'pytorch_model.bin'))
        model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)        
    
    
    # 1. pretrain a BERT in stage1
    if not LOAD_CHECKPOINT:
        for epoch in range(num_epochs):
            model.train()
            
            losses = []
            for i, batch in enumerate(train_loader):
                loss, _ = model(**batch)
                losses.append(accelerator.gather(loss.repeat(config.batch_size)))
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()  

            loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
            loss_valid = validate(model, val_loader, accelerator)
            accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}')                

            writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar('perplexity_valid', loss_valid, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
            
        accelerator.save_state(STORE_PATH)
        
    # 2. use pretrained BERT to generate fake data for stage2 replay
    
    layer_outputs = [[] for _ in range(config.num_hidden_layers)]
    layer_replay_data = {}    
    
    with torch.no_grad():
        # get layer outputs
        if ONLY_LAST_LAYER:
            for i, batch in enumerate(train_loader):
                print(i)
                if i >= SAMPLE_BATCHES:                     
                    break
                h_ = model.bert(batch['input_ids'], batch['attention_mask'])    
                layer_index = config.num_hidden_layers-1 
                            
                if i == 0:
                    layer_outputs[layer_index].append(h_.to('cpu'))            
                else:
                    layer_outputs[layer_index][0] = torch.cat([layer_outputs[layer_index][0], h_.to('cpu')], dim=0)   
                                                                    
        else:
            for i, batch in enumerate(train_loader):
                if i >= SAMPLE_BATCHES:
                    break
                h_ = model.bert.embeddings(batch['input_ids'])
                for l in range(NUM_REPLAY_LAYERS):
                    h_ = model.bert.encoders.layers[l](h_, batch['attention_mask'])
                    
                    if i == 0:
                        layer_outputs[l].append(h_.to('cpu'))
                    else:
                        layer_outputs[l][0] = torch.cat([layer_outputs[l][0], h_.to('cpu')], dim=0)
                            
                            
        # generate fake data (only input_ids)
        for i, layer_output in enumerate(layer_outputs):
            if not len(layer_output):
                continue
            sample_indexes = sample_by_cluster(layer_output[0].mean(dim=1), NUM_CLUSTERS, DATA_PER_CLUSTER)    
            sample_outputs = layer_output[0][sample_indexes]
            h_ = sample_outputs.to('cuda')
            
            for j in range(i + 1, config.num_hidden_layers):
                h_ = model.bert.encoders.layers[j](h_, torch.ones(h_.shape[0], h_.shape[1]).to('cuda'))
            
            scores = model.head(h_)
            input_ids = torch.argmax(scores, dim=2)
                        
            layer_replay_data[str(i)] = input_ids
                
        # save fake data (replay data)        
        torch.save(layer_replay_data, REPLAY_FILE_PATH)
 

if __name__ == '__main__':
    main()