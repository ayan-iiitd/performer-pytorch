## modules needed

import tqdm
import torch
import torch.nn as nn
import ast
import torch.optim as optim
from performer_pytorch import PerformerEncDec
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import pandas
import math
import joblib
import os

seed = 666
torch.manual_seed(seed)
torch.autograd.set_detect_anomaly(True)

print("Setting constants")
## constants

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
GENERATE_EVERY  = 100
NUM_TOKENS = 28996 + 2
# NUM_TOKENS = 30000
ENC_SEQ_LEN = 512
DEC_SEQ_LEN = 256 + 1
STEP_SIZE = 200000
START_STEP = 0
MODEL_PATH = './models'

print("Creating Dataset")
## create dataset

class SummaryDataset(Dataset):

    def __init__(self, filename):

        summary_data = pandas.read_csv(filename)
        
        x = summary_data['src_txt_tokens'].apply(ast.literal_eval)
        y = summary_data['tgt_txt_tokens'].apply(ast.literal_eval)
        xm = summary_data['src_txt_att_mask'].apply(ast.literal_eval)
        ym = summary_data['tgt_txt_att_mask'].apply(ast.literal_eval)

        self.X = torch.tensor(x)
        self.Y = torch.tensor(y)
        self.X_mask = torch.tensor(xm)
        self.Y_mask = torch.tensor(ym)
    

    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, index):
        
        src = self.X[index]
        src_msk = self.X_mask[index].bool()
        
        one = torch.ones(1)
        tgt = torch.cat((one, self.Y[index]), 0)
        tgt_msk = torch.cat((one, self.Y_mask[index]), 0).bool()
        
        return (src, src_msk, tgt, tgt_msk, index)


print("Creating Dataloader")
## create dataloader

#summary_dataset = SummaryDataset("/home/ayan/ayan_fed_home/data/python_files/my_summ_data/datasets/train_tokens.csv")
#joblib.dump(summary_dataset, "data/summary_train_dataset")
sm_dataset = joblib.load("data/summary_train_dataset")
summary_dataloader = DataLoader(sm_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = torch.get_num_threads())


print("Instantiating Model")
## instantiate model

model = PerformerEncDec(
    dim = 512,
    enc_num_tokens = NUM_TOKENS,
    enc_depth = 1,
    enc_heads = 8,
    enc_max_seq_len = ENC_SEQ_LEN,
    enc_reversible = True,
    enc_feature_redraw_interval = 1000,
    enc_nb_features = 64,
    dec_num_tokens = NUM_TOKENS,
    dec_depth = 3,
    dec_heads = 8,
    dec_max_seq_len = DEC_SEQ_LEN,
    dec_reversible = True,
    dec_feature_redraw_interval = 1000,
    dec_nb_features = 64
)

# model = nn.DataParallel(model)
model = model.cuda()


print("Setting Optimizer")
## optimizer

optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
scaler = GradScaler()


if any(os.scandir(MODEL_PATH)):

    #directory = "/home/ayan/data/python_files/my_summ_data/datasets/train"
    model_path = os.path.abspath(MODEL_PATH)
    model_list = [entry.path for entry in os.scandir(model_path) if entry.is_file()]
    iteration_completed_indices = []

    for file_index in range(len(model_list)):
        iteration_completed_indices.append(int(model_list[file_index].split('_--_')[1]))

    iteration_completed_indices.sort()
    final_model_path = os.path.join(MODEL_PATH, 'iter_--_', str(iteration_completed_indices[-1]), '_--_checkpoints.pt')
    print(iteration_completed_indices)
    print(final_model_path)
    model.load_state_dict(final_model_path['model_state_dict'])
    optim.load_state_dict(final_model_path['optimizer_state_dict'])

    
print("Starting Training")
## training model

for iteration in tqdm.tqdm(range(START_STEP, STEP_SIZE), mininterval = 10., desc = 'training'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    src, src_mask, tgt, tgt_mask, indices = next(iter(summary_dataloader))
    src_mask = src_mask.cuda()
    src = src.long().cuda()
    tgt_mask = tgt_mask.cuda()
    tgt = tgt.long().cuda()
    
    with autocast():
        loss = model(src, tgt, enc_mask = src_mask, dec_mask = tgt_mask)
    
    # print(src.shape, src_mask.shape, tgt.shape, tgt_mask.shape)
    # print("\n\n")
    # print(indices)
    # print(loss)
    # scaler.scale(loss).backward()
    scaler.scale(loss).sum().backward()
    
    # print(f'{i}: {loss.item()}')
    print(f'{iteration}: {loss.sum()}')

    scaler.step(optim)
    scaler.update()
    optim.zero_grad()

    if iteration%10 == 0:

        path = os.path.join(MODEL_PATH, 'iter_--_' + str(iteration) + '_--_checkpoints.pt')
        torch.save({
            'iteration' : iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
            }, path)

    if iteration != 0 and iteration % GENERATE_EVERY == 0:
        
        model.eval()
        src, src_mask, _, _, _ = next(iter(summary_dataloader))

        src, src_mask = src[:1], src_mask[:1]
        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

        src = src.cuda()
        src_mask = src_mask.cuda()

        sample = model.generate(src, start_tokens, ENC_SEQ_LEN, enc_mask = src_mask)
        # sample = model.module.generate(src, start_tokens, ENC_SEQ_LEN, enc_mask = src_mask)
        incorrects = (src != sample).abs().sum()

        print(f"input:  ", src)
        print(f"predicted output:  ", sample)
        print(f"incorrects: {incorrects}")