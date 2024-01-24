import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

import torch
import torch.utils.data as utils

from tqdm.auto import tqdm
from copy import deepcopy
import re

#re.sub(r'bed$', 'bim', bed_file)

# def read_snp(gwas_file, P_threshold=0.05):
def read_snp(gwas_file):
    ss = pd.read_csv(gwas_file, sep=' +', engine='python',names = ['SNP'])
    # ss = pd.read_csv(gwas_file,
                     # sep='\s+', engine='python',
                     # usecols=['SNP','TEST' ,'OR','P'],
                     # index_col='SNP')
    # ss = ss.loc[ss.TEST == 'ADD']
    # is_significant =  ss['P']<=P_threshold
    # sig_snps = ss[is_significant].index.tolist()
    sig_snps = ss.SNP.tolist()

    return sig_snps


def load_dataset(genotype_file, sample_file, *, train_sample_file=None):
    sample_all = pd.read_csv(sample_file)
                      
                      
    samples = pd.read_csv(train_sample_file,sep=' ').FID.tolist()

            
    train_sample_idx = np.flatnonzero(sample_all.FID.isin(samples))
     
    test_sample_file = re.sub('train','test',train_sample_file)
    val_sample_file = re.sub('train','val',train_sample_file)     
        
    #list_id = except_str.split('-')
    #sample_file = re.sub(r'merge','sample',train_sample_file)
    #sample_file = re.sub(r'samples','sample',sample_file)
    #sample_file = re.sub(r'except','list',sample_file)
    #test_sample_file = re.sub(except_str,list_id[0],sample_file)
    #val_sample_file = re.sub(except_str,list_id[1],sample_file)
    
    
    samples = pd.read_csv(test_sample_file,sep=' ').FID.tolist()
    test_sample_idx = np.flatnonzero(sample_all.FID.isin(samples))

    samples = pd.read_csv(val_sample_file,sep=' ').FID.tolist()
    val_sample_idx = np.flatnonzero(sample_all.FID.isin(samples))

    skipcols = ["FID","PHENO"]
    data = pd.read_csv(genotype_file, usecols=lambda x: x not in skipcols).to_numpy()
    
    col_idx = range(0, data.shape[1])
    x_train = data[np.ix_(train_sample_idx, col_idx)]

    # Convert case/contral encoding from 'PLINK style' {1:contral, 2:case} to {0:contral, 1:case}
    y_train = sample_all.PHENO.iloc[train_sample_idx].values - 1
    
    x_val = data[np.ix_(val_sample_idx, col_idx)]

    # Convert case/contral encoding from 'PLINK style' {1:contral, 2:case} to {0:contral, 1:case}
    y_val = sample_all.PHENO.iloc[val_sample_idx].values - 1
    
    x_test = data[np.ix_(test_sample_idx, col_idx)]

    # Convert case/contral encoding from 'PLINK style' {1:contral, 2:case} to {0:contral, 1:case}
    y_test = sample_all.PHENO.iloc[test_sample_idx].values - 1
    
    
    

    return x_train, y_train, x_val, y_val, x_test, y_test
    
    
    
    
    
    
    
    
    


def permute(x, y):
    if len(x) != len(y):
        raise ValueError('x and y should have same length in the first dimension.')

    p = np.random.permutation(len(x))
    return x[p], y[p]

def split(data, *, val_ratio):
    split = int(len(data) * (1 - val_ratio))
    return data[:split], data[split:]

def arr_to_dataloader(x, y, batch_size=10, **kargs):
    dataset = [torch.from_numpy(d).float() for d in (x, y)]
    dataset = utils.TensorDataset(*dataset)
    loader = utils.DataLoader(dataset, batch_size=batch_size, **kargs)
    return loader


def accuracy_from_score(pred, true):
    pred = (pred > 0.5).float()
    return pred.eq(true).sum().float().item()


## Model fitting and evaluation
def foward_batches(loader, model, loss_fn, opti=None):
    n = 0
    loss = 0
    acc = 0

    pred = []
    true = []
    auroc = 0

    for _, batch in tqdm(enumerate(loader), total=len(loader)):
        x, y = batch
        x, y = x.cuda(), y.cuda()
        n += y.size(0)

        out = model(x)
        _loss = loss_fn(out, y.view_as(out))

        if opti is not None:
            opti.zero_grad()
            _loss.backward()
            opti.step()

        out, y = out.detach().squeeze(), y.detach().squeeze()
        loss += _loss.item()
        acc += accuracy_from_score(out, y)
        pred.append(out.cpu().numpy())
        true.append(y.cpu().numpy())

    auroc = roc_auc_score(np.concatenate(true),
                          np.concatenate(pred))

    return loss/n, acc/n, auroc


def fit(model, loss_fn, optimizer, *,
        epochs,
        train_loader, val_loader, test_loader=None,
        model_dir=None):

    best_auroc = 0.5

    performance = pd.DataFrame(columns = ['epoch', 'stage', 'loss', 'acc', 'auroc'])
    for epoch in range(epochs):
        #Train
        model.train()

        train_loss, \
        train_acc, \
        train_auroc = foward_batches(train_loader, model, loss_fn, optimizer)
        print(f'Epoch: {epoch}, ',
              f'Train Loss: {train_loss:4f}, ',
              f'Train Acc: {train_acc:4f}, ',
              f'Train AUROC: {train_auroc:4f}')

        row = {'epoch': epoch, 'stage': 'train',
               'loss': train_loss, 'acc': train_acc, 'auroc': train_auroc}
        #performance = performance.append(row, ignore_index=True)
        performance = pd.concat([performance,pd.DataFrame([row])], ignore_index=True)

        
        if model_dir is not None:
            torch.save(model.state_dict(), f'{model_dir}/epoch{epoch}.pt')
        ##Val
        # In order to do inference, we need to put the network in eval mode.
        # If we forget to do that, some pretrained models, like batch normalization 
        # and dropout, will not produce meaningful answers.
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_auroc = foward_batches(val_loader, model, loss_fn)
            print(f'Epoch: {epoch}, ',
                  f'Validation Loss: {val_loss:4f}, ',
                  f'Validation Acc: {val_acc:4f}, ',
                  f'Validation AUROC: {val_auroc:4f}')
        row = {'epoch': epoch, 'stage': 'validation',
               'loss': val_loss, 'acc': val_acc, 'auroc': val_auroc}
        #performance = performance.append(row, ignore_index=True)
        performance = pd.concat([performance,pd.DataFrame([row])], ignore_index=True)

        if val_auroc > best_auroc or epoch == 0:
            best_model_state = deepcopy(model.state_dict())
            best_auroc = val_auroc

        #Test
        if test_loader is not None:
            with torch.no_grad():
                test_loss, test_acc, test_auroc = foward_batches(test_loader, model, loss_fn)
                print(f'Epoch: {epoch}, ',
                      f'Test Loss: {test_loss:4f}, ',
                      f'Test Acc: {test_acc:4f}, ',
                      f'Test AUROC: {test_auroc:4f}')
            row = {'epoch': epoch, 'stage': 'test',
                   'loss': test_loss, 'acc': test_acc, 'auroc': test_auroc}
            #performance = performance.append(row, ignore_index=True)
            performance = pd.concat([performance,pd.DataFrame([row])], ignore_index=True)

    return performance, best_model_state
    
    
def predict(loader, model):
    pred = []
    true = []

    model.eval()
    with torch.no_grad():
        for _, batch in tqdm(enumerate(loader), total=len(loader)):
            x, y = batch
            x, y = x.cuda(), y.cuda()

            out = model(x)

            out, y = out.detach().squeeze(), y.detach().squeeze()
            pred.append(out.cpu().numpy())
            true.append(y.cpu().numpy())
    true = np.concatenate(true)
    pred = np.concatenate(pred)

    return pred, true
