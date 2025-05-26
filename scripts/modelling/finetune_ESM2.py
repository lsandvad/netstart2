#!/usr/bin/env python3.8

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from transformers import AutoTokenizer, EsmForSequenceClassification

#Clear the GPU memory cache
torch.cuda.empty_cache()

#Configure CUDA memory allocations (helps manage fragmentation in the GPU memory)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

#Define the model (train/val partitions) to finetune
model_no = 3
esm_model = "facebook/esm2_t6_8M_UR50D"
esm_model_title = "esm2-8m"


#define number of aas to extract
extract_upstream_aa = 100
extract_downstream_aa = 100

def encode_nucleotide_to_amino_acid(sequence):
    """
    The function takes a nucleotide sequence and translates it into the corresponding amino acid sequence.
    The genetic code dictionary maps each three-nucleotide codon to its corresponding amino acid. 
    Stop codons are represented by '<unk>' and padding codons are represented by '<pad>'.

    Args:
        sequence (str): A nucleotide sequence.

    Returns:
        amino_acid_sequence (str): The corresponding amino acid sequence.
    """
    #Define the genetic code as a dict
    genetic_code = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '<unk>', 'TAG': '<unk>',
        'TGT': 'C', 'TGC': 'C', 'TGA': '<unk>', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }

    #Ensure the sequence length is a multiple of 3
    assert len(sequence) % 3 == 0, "Input sequence length must be a multiple of 3"

    #Initialize
    amino_acid_sequence = ""

    #Iterate through the nucleotide sequence to encode codons
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        amino_acid = genetic_code.get(codon, "<pad>") 
        amino_acid_sequence += amino_acid

    return amino_acid_sequence


def extract_datasets(data_train,
                     data_val,
                     extract_upstream_aa,
                     extract_downstream_aa):
    """
    Create the datasets to use for model development.

    Args:
        data_train (df): the loaded dataframe with train partitions
        data_val (df): the loaded dataframe with validation partition
        extract_upstream_aa (int): the number of amino acid positions upstream ATG to extract
        extract_downstream_aa (int): the number of amino acid positions downstream ATG to extract

    Returns:
        aa_sequences_train (list): the amino acid sequences for the training set
        labels_train (list): the labels for the training set
        aa_sequences_val (list): the amino acid sequences for the validation set
        labels_val (list): the labels for the validation set
    """

    #Initialize
    aa_sequences_train = []     #x train
    labels_train = []           #y train

    aa_sequences_val = []       #x validate
    labels_val = []             #y validate

    #Initialize
    ATG_position = 300 #we extracted 300 nt upstream for dataset

    #Extract sequences from training dataset
    for row in data_train.itertuples():

        assert row.Sequence[ATG_position:ATG_position+3] == "ATG"

        #Get specificed amino acid subsequence
        aa_sequences_train.append(encode_nucleotide_to_amino_acid(row.Sequence[ATG_position-3*extract_upstream_aa:ATG_position+3+3*extract_downstream_aa]))

        #Get labels (TIS or non-TIS)
        labels_train.append((row.TIS))


    #Extract sequences from validation dataset
    for row in data_val.itertuples():

        assert row.Sequence[ATG_position:ATG_position+3] == "ATG"

        aa_sequences_val.append(encode_nucleotide_to_amino_acid(row.Sequence[ATG_position-3*extract_upstream_aa:ATG_position+3+3*extract_downstream_aa]))
        labels_val.append(float(row.TIS))

    return aa_sequences_train, labels_train, aa_sequences_val, labels_val


class SeqDataset(torch.utils.data.Dataset):
    """
    This class is used to create a dataset for the model

    Args:
        encodings (dict): the encoded amino acid sequences
        labels (list): the labels for the sequences

    Returns:
        item (dict): the encoded amino acid sequences and labels
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.astype(np.float32)  # Convert labels to float32

    def __getitem__(self, idx):
        # Use .clone() or .detach() for tensor copying
        item = {key: val[idx].clone() for key, val in self.encodings.items()}
        item['labels'] = torch.as_tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


###Main code###

#Read the compressed CSV files for each partition into df
print("Reading compressed data in partitions")
data_partition_1 = pd.read_csv("../../data/data_model/datasets/data_partition_1_masked.csv.gz", 
							   dtype={'annotation_source': 'str'},
							   compression='gzip')
data_partition_2 = pd.read_csv("../../data/data_model/datasets/data_partition_2_masked.csv.gz", 
							   dtype={'annotation_source': 'str'},
							   compression='gzip')
data_partition_3 = pd.read_csv("../../data/data_model/datasets/data_partition_3_masked.csv.gz", 
							   dtype={'annotation_source': 'str'},
							   compression='gzip')
data_partition_4 = pd.read_csv("../../data/data_model/datasets/data_partition_4_masked.csv.gz", 
							   dtype={'annotation_source': 'str'},
							   compression='gzip')

#Extract datasplits for train and valiation sets for the different models
print("Defining training/validation split for model")
if model_no == 1:
    #Concat train partitions
    data_train = pd.concat([data_partition_2,
                            data_partition_3,
                            data_partition_4],
                           ignore_index = True)
    data_val = data_partition_1

elif model_no == 2:
    #Concat train partitions
    data_train = pd.concat([data_partition_1,
                            data_partition_3,
                            data_partition_4],
                           ignore_index = True)
    data_val = data_partition_2

elif model_no == 3:
    #Concat train partitions
    data_train = pd.concat([data_partition_1,
                            data_partition_2,
                            data_partition_4],
                           ignore_index = True)
    data_val = data_partition_3

if model_no == 4:
    #Concat train partitions
    data_train = pd.concat([data_partition_1,
                            data_partition_2,
                            data_partition_3],
                           ignore_index = True)
    data_val = data_partition_4


print(data_train.shape)
print(data_val.shape)
print(data_train)

print("Extracting datasets (sequences and labels separately)")
aa_sequences_train, labels_train, \
aa_sequences_val, labels_val = extract_datasets(data_train,
                                                  data_val,
                                                  extract_upstream_aa,
                                                  extract_downstream_aa)


#Check that the lengths of the sequences and labels are the same
assert len(aa_sequences_train) == len(labels_train)
assert len(aa_sequences_val) == len(labels_val)

print("Fraction of TIS-labelled samples in training data: ", round(sum(labels_train)/len(labels_train), 3))

#Encode labels to ensure correct format
encoder = LabelEncoder()
encoder.fit(labels_train)
labels_train = encoder.transform(labels_train)
encoder.fit(labels_val)
labels_val = encoder.transform(labels_val)

#Define the maximum length of the amino acid sequences
aa_seqs_len = extract_upstream_aa + 1 + extract_downstream_aa

#Load tokenizer
tokenizer_aa = AutoTokenizer.from_pretrained(esm_model, 
                                             do_lower_case=False, 
                                             model_max_length=aa_seqs_len + 2) # CLS token + aa sequence (1 token each) + EOS token


#Turn amino acid sequences into tokens for model input
train_encodings_aa = tokenizer_aa(aa_sequences_train, 
                            padding=True,  #pad sequences to max length and apply attention mask
                            truncation=True, 
                            return_tensors="pt")

#Create attention mask
attention_mask_train = train_encodings_aa['input_ids'] != tokenizer_aa.pad_token_id
train_encodings_aa['attention_mask'] = attention_mask_train.int()  # Replace the old attention mask

val_encodings_aa = tokenizer_aa(aa_sequences_val, 
                          padding=True,  #pad sequences to max length and apply attention mask
                          truncation=True,
                          return_tensors="pt")
attention_mask_val = val_encodings_aa['input_ids'] != tokenizer_aa.pad_token_id
val_encodings_aa['attention_mask'] = attention_mask_val.int()  # Replace the old attention mask

aa_encodings_len = train_encodings_aa['input_ids'].shape[1] #amino acids + CLS + EOS

#Get datasets in formats that can be fed into model
train_dataset = SeqDataset(train_encodings_aa, labels_train)
val_dataset = SeqDataset(val_encodings_aa, labels_val)

#Define model and run on GPU when possible
device = torch.device("mps" if torch.mps.is_available() else "cpu")

print("Running on: ", device, flush = True)
print(f"Memory Allocated before loading model: {torch.cuda.memory_allocated(device) / 1024**3} GB")
model = EsmForSequenceClassification.from_pretrained(esm_model, num_labels=1)
model.to(device)
print(f"Memory Allocated after loading model: {torch.cuda.memory_allocated(device) / 1024**3} GB")

# If multiple GPUs are available, use DataParallel
#if torch.cuda.is_available() and torch.cuda.device_count() > 1:
#    print(f"Using {torch.cuda.device_count()} GPUs!")
#    model = torch.nn.DataParallel(model)  # This wraps your model to use multiple GPUs

#Define settings for training
epochs = 4
batch_size = 32
steps_per_epoch = len(train_dataset) / (batch_size)                          #The number of steps taken per epoch
eval_per_epoch = 10                                                          #The number of evaluations made per epoch
steps_between_evals = int(steps_per_epoch / eval_per_epoch)                  #The number of steps to take between each evaluation
total_eval_steps = eval_per_epoch * epochs                                   #The total number of times the model is evaluated during finetuning

print("Running for ", epochs, " epochs with a batchsize of ", batch_size, \
      ". Total number of training sequences: ", len(train_dataset), sep = "")
print("Evaluates every ", steps_between_evals, " steps.", sep = "")


#Define data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,   # Shuffle the data
    num_workers=8,  # Adjust based on your CPU cores
    pin_memory=True # Optimizes data transfer to GPU
    )

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=8,  
    pin_memory=True
    )

#include class weights in loss function
pos_weight = torch.tensor(3, dtype=torch.float).to(device)

#Define the binary cross-entropy loss criterion
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-6)

#Loss across epochs
loss_train_steps = []
loss_val_steps = []
step_number = []
epoch_number = []

#Initialize variables for early stopping
best_val_loss = float('inf')
threshold_patience = 6  #Number of evaluations with no improvement to wait before stopping
counter_patience = 0

#Initialize variables for training loop
train_loss = 0.0
step = 0

print("Initializing training", flush = True)
#Training loop
for epoch in range(epochs):
    print("Epoch: ", epoch+1, flush=True)
    
    #Iterate through the training data batch-wise
    for i, batch in enumerate(train_loader):
        
        #Set model to training mode
        model.train()

        #Get inputs and labels
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch['labels'].to(device).view(-1, 1).float()

        #Clear the GPU memory cache
        torch.cuda.empty_cache() 

        #Forward pass
        outputs = model(inputs, attention_mask)

        #Calculate loss
        loss = loss_fn(outputs.logits, labels)
        
        #Backward pass
        optimizer.zero_grad()   # Clear gradients
        loss.backward()         # Calculate gradients
        
        #Update parameters
        optimizer.step()

        train_loss += loss.item()
        step += 1

        torch.cuda.empty_cache()
        del outputs
        del loss
        
        if step % int(steps_between_evals/4) == 0:
            print(f"Step {step}", sep = " ", flush=True)
        
        #Make evaluation of model
        if step % steps_between_evals == 0:
            #Set model to evaluation mode
            model.eval()

            val_loss = 0.0
            val_true_labels = []
            val_predicted_probs = []

            #Iterate through the validation data batch-wise
            with torch.no_grad():
                for counter, batch in enumerate(val_loader):
                    #Get inputs and labels
                    inputs = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch['labels'].to(device).view(-1, 1).float()

                    #Forward pass
                    outputs = model(inputs, attention_mask)
                    loss = loss_fn(outputs.logits, labels)

                    probabilities = torch.sigmoid(outputs.logits)

                    val_loss += loss.item()
                    
                    val_true_labels.extend(labels.cpu().numpy().flatten())
                    val_predicted_probs.extend(probabilities.cpu().numpy().flatten())

                #Calculate metrics
                train_avg_loss = round(train_loss / steps_between_evals, 4)
                val_avg_loss = round(val_loss / len(val_loader), 4)
                val_auc_roc = round(roc_auc_score(val_true_labels, val_predicted_probs), 4)
                precision, recall, _ = precision_recall_curve(val_true_labels, val_predicted_probs)
                val_pr_auc = round(auc(recall, precision), 4)

                del loss
                del outputs
                torch.cuda.empty_cache()
                
                #Store loss progression
                loss_train_steps.append(train_avg_loss)
                loss_val_steps.append(val_avg_loss)
                step_number.append(step)
                epoch_number.append(epoch+1)
                
                #Re-initialize
                train_loss = 0.0

                #Print metrics
                print(f"Train Loss: {train_avg_loss}\tVal Loss: {val_avg_loss}\tVal AUC ROC: {val_auc_roc}\tVal AUC PR: {val_pr_auc}",
                      sep = "", flush=True)
                
                ##Check validation loss, apply early stopping if necessary
                if val_avg_loss < best_val_loss:
                    
                    #Update lowest loss
                    best_val_loss = val_avg_loss
                    
                    #Reset counter
                    counter_patience = 0

                    #Get the state dict (strip 'module.' if needed)
                    #if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    #    model = model.module.state_dict()  # Strips 'module.' prefix
                    #else:
                    #    model = model.state_dict()

                    #Save the fine-tuned LM layers without the classification head
                    base_model = model.base_model

                    #Save only the LM layers without the classification head
                    base_model.save_pretrained('../../data/data_model/pretrained_models/finetuned_models/'+esm_model_title+'-finetuned_model_100u_100d_model'+str(model_no))
                    
                    # Save the fine-tuned model with classification head
                    model.save_pretrained('../../data/data_model/models/' + esm_model_title + "_finetuned_full_model_100u_100d_model" + str(model_no) + ".pth")
        
                else:
                    counter_patience += 1
                    
            #When the model is overfitting, use early stopping
            if counter_patience >= threshold_patience:
                print("Early stopping. No improvement in validation loss.", flush=True)
                break