import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
import torch.utils.data as utils

from tqdm.auto import tqdm
from .PairNetClassifier import PairNetClassifier
#from model import Model
#from src.model import Model
from .utils_temp import (
       load_dataset,
       permute,
       split,
       arr_to_dataloader,
       accuracy_from_score,
       foward_batches,
       fit,
       predict
)
import os



################################################################################
#
# Main
#
################################################################################
def train(train_z, train_p, val_z, val_p, test_z, test_p,OUT_DIR,SEED=None,EPOCHS = 20,MODEL_DIR = None,log_file = None):
    """train the data
    """
    if SEED is not None:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # use deterministic algorithms
    # torch.use_deterministic_algorithms(True)  
    torch.backends.cudnn.benchmark = False


    # Load datasets

    #train_z, train_p, val_z, val_p, test_z, test_p = load_dataset(genotype_file, sample_file,
    #                             train_sample_file=train_id_file )
    n_feature = train_z.shape[1]


    if train_z.shape[1] != val_z.shape[1] or train_z.shape[1] != test_z.shape[1]:
        print("The number of columns in the matrices is inconsistent, cannot perform the operation.")
        exit(1)
    if train_z.shape[0] != len(train_p) or val_z.shape[0] != len(val_p) or test_z.shape[0] != len(test_p):
        print("The number of rows in the matrices is inconsistent, cannot perform the operation.")
        exit(1)
    p = np.concatenate([train_p,val_p,test_p])
    if np.any((p != 1) & (p != 2)):
        if np.any((p != 0) & (p != 1)):
            print("Please represent traits as 0, 1, or 1, 2.")
            exit(1)
    else:
        train_p = train_p - 1
        val_p = val_p - 1
        test_p = test_p - 1
    # geno, pheno = permute(geno, pheno)
    # train_z, val_z = split(geno, val_ratio = 0.2)
    # train_p, val_p = split(pheno, val_ratio = 0.2)

    #if n_feature == 0:
    #    with open(log_file,"a") as f:
    #        f.write("Number of features %i \n" %n_feature)
    #    with open(snakemake.output.roc,"a") as f:
    #        f.write("No plot")
    #    with open(snakemake.output.train,"a") as f:
    #        f.write("No plot")
    #    for model in snakemake.output.models:
    #        with open(model,"a") as f:
    #            f.write("No model")
    #    performance =  pd.DataFrame({"epoch":[0,0],	
    #                                "stage":["validation","test"],	
    #                                "loss":[0,0],	
    #                                "acc":[0,0],	
    #                                "auroc":[0,0]})
    #    performance.to_csv(snakemake.output.performance, index=False)
    #    exit()
    if log_file is not None:
        with open(log_file,"a") as f:
            f.write("Number of features %i \n" %n_feature)
            f.write("data-------------------------\n")
            f.write("number of samples in data: %i\n" %(len(train_p)+len(val_p)+len(test_p)) )
            f.write("number of case in data: %i\n" %(sum(train_p == 1)+sum(val_p == 1)+sum(test_p == 1)))
            f.write("number of contral in data: %i\n" %(sum(train_p == 0)+sum(val_p == 0)+sum(test_p == 0)))
            f.write("train-------------------------\n")
            f.write("number of samples in training data: %i\n" %len(train_p))
            f.write("number of case in training data: %i\n" %sum(train_p == 1))
            f.write("number of contral in training data: %i\n" %sum(train_p == 0))
            f.write("validation--------------------\n")
            f.write("number of samples in validation data: %i\n" %len(val_p))
            f.write("number of case in validation data: %i\n" %sum(val_p == 1))
            f.write("number of contral in validation data: %i\n" %sum(val_p == 0))
            f.write("test--------------------------\n")
            f.write("number of samples in testing data: %i\n" %len(test_p))
            f.write("number of case in testing data: %i\n" %sum(test_p == 1))
            f.write("number of contral in testing data: %i\n" %sum(test_p == 0))
            f.write("------------------------------\n")
    # Convert data to torch dataset format
    train_loader = arr_to_dataloader(train_z, train_p, batch_size=256, shuffle=True)
    val_loader = arr_to_dataloader(val_z, val_p, batch_size=256, shuffle=False)
    test_loader = arr_to_dataloader(test_z, test_p, batch_size=256, shuffle=False)


    # Train
    # torch.cuda.set_device(GPU_ID)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = PairNetClassifier(n_feature).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    loss_fn = F.binary_cross_entropy

    performance , best_model_dict= fit(model, loss_fn, optimizer,
                      epochs=EPOCHS,
                      train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                      model_dir=MODEL_DIR)

    performance.to_csv(OUT_DIR+'_performance.csv', index=False)


    # Plot
    # best_epoch = (
        # performance[performance.stage == 'validation']
                   # [lambda df: df.auroc == df.auroc.max()].epoch.values[0]
    # )

    # best_model = Model(n_feature).cuda()
    # best_model.load_state_dict(torch.load(f'{MODEL_DIR}/epoch{best_epoch}.pt'))
    # torch.save(best_model.state_dict(), f'{MODEL_DIR}/epoch{best_epoch}_best_model.pt')
    best_model = PairNetClassifier(n_feature).cuda()
    best_model.load_state_dict(best_model_dict)
    ## Training process
    fig_proc, ax_proc = plt.subplots()
    # ax_proc.axvline(x=best_epoch, color='red')
    for label, df in performance.groupby('stage'):
        ax_proc.plot(df.epoch.astype('int'), df.auroc, label=label)

    ax_proc.legend()
    ax_proc.set_title('Training Process')
    ax_proc.set_xlabel('epoch')
    ax_proc.set_ylabel('AUROC')

    fig_proc.tight_layout()
    
    fig_proc.savefig(OUT_DIR+'_training.jpeg')

    ## ROC
    val_pred, val_true = predict(val_loader, best_model)
    val_fpr, val_tpr, val_thresholds = roc_curve(val_true, val_pred)
    val_auroc = roc_auc_score(val_true, val_pred)


    test_pred, test_true = predict(test_loader, best_model)
    test_fpr, test_tpr, test_thresholds = roc_curve(test_true, test_pred)
    test_auroc = roc_auc_score(test_true, test_pred)

    fig_auroc, ax_auroc = plt.subplots()
    ax_auroc.plot([0, 1], [0, 1], linestyle='--')
    ax_auroc.plot(val_fpr, val_tpr, marker='.', label = 'validation')
    ax_auroc.plot(test_fpr, test_tpr, marker='.', label = 'test')

    ax_auroc.legend()
    ax_auroc.set_title(f'AUROC val:{val_auroc:.3f}  test:{test_auroc:.3f}')
    ax_auroc.set_ylabel('True Positive Rate')
    ax_auroc.set_xlabel('False Positive Rate')

    fig_auroc.tight_layout()
    fig_auroc.savefig(OUT_DIR+'_roc.jpeg')
