"""
Pretrain Stage2:
1. Normal Continual Pretrain
2. Train with stage1 data replay
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
from Dataset import ACLForLM, ReplayDataset, MixedData
from utils import *


"""
WARNING: Names/Parameters That You Should Keep The Same as That in Pretrain Stage 1
1. LOAD_FOLDER: should be same as STORE_FOLDER in stage 1
2. REPLAY_FILE: should be same as REPLAY_FILE in stage 1
3. NUM_REPLAY_LAYERS: should be same as NUM_REPLAY_LAYERS in stage 1 (especially if you choose to replay but not only on last layer)
4. ONLY_LAST_LAYER: should be same as ONLY_LAST_LAYER in stage 1 (MUST set this to be same in stage 1 if you want to correctly replay)

Also, Remember to set REPLAY=True/False, when you do/do not want to replay
"""


# replay config
REPLAY = False
NUM_REPLAY_LAYERS = 12
ONLY_LAST_LAYER = False

# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder paths
LOAD_FOLDER = "1103-stage1"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
STORE_FOLDER = "1103-stage2-no-replay"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
REPLAY_FILE = 'replay-data-12s-8*8.pth'
REPLAY_FILE_PATH = os.path.join(LOAD_PATH, REPLAY_FILE)
CONFIG_PATH = 'config/bert.json'


# training parameters
replay_steps = 1
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

    dataset = ACLForLM(config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    old_dataset = MixedData(config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    
    model = base_models.BertForMLM(config)    
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    old_val_loader = old_dataset.val_loader
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    model, optimizer, lr_scheduler, train_loader, val_loader, old_val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, old_val_loader)        
    
    if REPLAY:
        dataset_replay = ReplayDataset(batch_size=config.batch_size, path=REPLAY_FILE_PATH)
        replay_loader = {}
        for key, loader in dataset_replay.replay_loader.items():  
            print(key)          
            replay_loader[key] = accelerator.prepare(loader)      
    

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
        
        if REPLAY:
            if epoch % replay_steps == 0:
                if ONLY_LAST_LAYER:
                    for j, batch in enumerate(replay_loader[str(config.num_hidden_layers-1)]):
                        loss, _ = model(**batch)                                        
                        optimizer.zero_grad()
                        accelerator.backward(loss)
                        optimizer.step()     
                else:
                    for l in range(NUM_REPLAY_LAYERS):
                        for j, batch in enumerate(replay_loader[str(l)]):
                            loss, _ = model(**batch)                                        
                            local_optimizer = optim.AdamW(model.bert.encoders.layers[l].parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)             
                            accelerator.backward(loss)
                            local_optimizer.step()     
                
                                                         
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator)
        loss_valid_old = validate(model, old_val_loader, accelerator)
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Old Valid Loss: {loss_valid_old}')                

        writer.add_scalar('loss_train', loss_train, epoch)
        writer.add_scalar('loss_valid', loss_valid, epoch)
        writer.add_scalar('loss_valid_old', loss_valid_old, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    accelerator.save_state(STORE_PATH)
        

if __name__ == '__main__':
    main()