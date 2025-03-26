import os
import numpy as np
import pandas as pd
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from collections import defaultdict

from transformers import AutoTokenizer, AutoModel

import wandb

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

#train all 4 models with the optimized, respective hyperparameters
model_no = 1
hyperpars_filename = 'netstart_model'+str(model_no)+'_hyperparameters.json'
esm2_pretrained = "esm2-8m-finetuned_model_100u_100d_model" + str(model_no)
esm2_model_filename = "esm2_model" + str(model_no)

#Collect optimized hyperparameters
with open("../../data/data_model/hyperparameters/"+hyperpars_filename, 'r') as json_file:
    model_config = json.load(json_file)

#Define hyperparameters
batch_size = model_config["batch_size"]
dropout_rate_1 = model_config["dropout_rate_1"]
lr = model_config["lr"]

#Define amino acid input window
aa_upstream = 100
aa_downstream = 100

# Define lists of species by category
fungi_list = ['Saccharomyces cerevisiae', 'Ustilago maydis', 'Schizosaccharomyces pombe',
              'Aspergillus nidulans', 'Cryptococcus neoformans', 'Neurospora crassa', 
              'Coprinopsis cinerea', 'Rhizophagus irregularis', 'Schizophyllum commune']

protozoa_list = ['Plasmodium falciparum', 'Entamoeba histolytica', 'Dictyostelium discoideum',
                 'Giardia intestinalis', 'Trypanosoma brucei', 'Leishmania donovani', 
                 'Toxoplasma gondii', 'Eimeria maxima']

plants_list = ['Oryza sativa', 'Arabidopsis thaliana', 'Selaginella moellendorffii', 'Brachypodium distachyon', 
               'Setaria viridis', 'Zea mays', 'Hordeum vulgare', 'Triticum aestivum', 
               'Phoenix dactylifera', 'Lotus japonicus', 'Medicago truncatula', 
               'Nicotiana tabacum', 'Glycine max', 'Solanum lycopersicum']

invertebrates_list = ['Trichoplax adhaerens', 'Tribolium castaneum', 'Manduca sexta', 
                      'Apis mellifera', 'Strongylocentrotus purpuratus', 'Daphnia carinata', 
                      'Drosophila melanogaster', 'Anopheles gambiae', 'Caenorhabditis elegans']

vertebrates_list = ['Gallus gallus', 'Alligator mississippiensis', 'Xenopus laevis',
                    'Oreochromis niloticus', 'Homo sapiens', 'Bos taurus', 'Mus musculus', 
                    'Ovis aries', 'Canis lupus', 'Equus caballus', 'Gorilla gorilla', 
                    'Pan troglodytes', 'Rattus norvegicus', 'Oryctolagus cuniculus', 'Sus scrofa', 
                    'Danio rerio', 'Oryzias latipes', 'Taeniopygia guttata', 'Columba livia', 
                    'Anolis carolinensis']


def assign_category(species):
    if species in fungi_list:
        return "fungi"
    elif species in protozoa_list:
        return "protozoa"
    elif species in plants_list:
        return "plant"
    elif species in invertebrates_list:
        return "invertebrate"
    elif species in vertebrates_list:
        return "vertebrate"

def load_and_partition_data(model_no):
    """
    Load in data and create train, validation and test splits
    
    Args:
        model_no (int [1, 2, 3, 4]): We are training 4 models; model no. corresponds to defining which partitions are train and which is validation
    """
    
    #Read the compressed CSV files for each partition into dfs
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
    data_partition_5 = pd.read_csv("../../data/data_model/datasets/data_partition_5_masked.csv.gz", 
                                   dtype={'annotation_source': 'str'},
                                   compression='gzip')

    #Extract datasplits for train and valiation sets for the different models to train
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

    if model_no == 3:
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

    #always extract partition 5 as test partition
    data_test = data_partition_5
    
    #Replace 'Canis lupus familiaris' with 'Canis lupus'
    data_train['Species'] = data_train['Species'].replace('Canis lupus familiaris', 'Canis lupus')
    data_val['Species'] = data_val['Species'].replace('Canis lupus familiaris', 'Canis lupus')
    data_test['Species'] = data_test['Species'].replace('Canis lupus familiaris', 'Canis lupus')

    data_train['group'] = data_train['Species'].apply(assign_category)
    data_val['group'] = data_val['Species'].apply(assign_category)
    data_test['group'] = data_test['Species'].apply(assign_category)

    # Sort the dataframe by the "group" column in descending order
    data_test = data_test.sort_values(by='group', ascending=False)

    return data_train, data_val, data_test


def encode_nucleotide_to_amino_acid(sequence):
    """
    The function takes a nucleotide sequence and translates it into the 
    corresponding amino acid sequence
    
    Args:
        sequence (str): A nucleotide sequence
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
                     data_test,
                     extract_upstream_aa,
                     extract_downstream_aa):
    """
    Create the datasets to use for model development.
    
    Args:
        data_train (df): the loaded dataframe with train partitions
        data_val (df): the loaded dataframe with validation partition
        data_test (df): the loaded dataframe with test partition
    """

    ATG_position = 300
    
    rows = []

    for i, row in enumerate(data_train.itertuples()):
        assert row.Sequence[ATG_position:ATG_position+3] == "ATG", print(row.Sequence[ATG_position:ATG_position+3])

        #Append each row of wanted information to the list
        rows.append({
            'aa_sequences_train': encode_nucleotide_to_amino_acid(row.Sequence[ATG_position-3*extract_upstream_aa:ATG_position+3+3*extract_downstream_aa]),
            'group_train': row.group,
            'labels_train': int(row.TIS)
        })
    
    #Store data in df
    df_train = pd.DataFrame(rows)
    
    rows = []
    for i, row in enumerate(data_val.itertuples()):
        assert row.Sequence[ATG_position:ATG_position+3] == "ATG", print(row.Sequence[ATG_position:ATG_position+3])
        
        rows.append({
            'aa_sequences_val': encode_nucleotide_to_amino_acid(row.Sequence[ATG_position-3*extract_upstream_aa:ATG_position+3+3*extract_downstream_aa]),
            'group_val': row.group,
            'labels_val': int(row.TIS)
        })
            
    df_val = pd.DataFrame(rows)
    
    rows = []
    for i, row in enumerate(data_test.itertuples()):
        
        assert row.Sequence[ATG_position:ATG_position+3] == "ATG", print(row.Sequence[ATG_position:ATG_position+3])

        rows.append({
            'aa_sequences_test': encode_nucleotide_to_amino_acid(row.Sequence[ATG_position-3*extract_upstream_aa:ATG_position+3+3*extract_downstream_aa]),
            'group_test': row.group,
            'labels_test': int(row.TIS)
        })

    df_test = pd.DataFrame(rows)

    return df_train, df_val, df_test

def create_encodings(df_train, df_val, df_test, extract_upstream_aa, extract_downstream_aa):
    """
    Encode labels and sequences to fit input to pretrained models.

    Args:
        df_train (dataframe): dataframe with training data
        df_val (dataframe): dataframe with validation data
        df_test (dataframe): dataframe with test data
    """
    
    #Encode labels to ensure correct format
    encoder = LabelEncoder()
    encoder.fit(df_train['labels_train'])
    labels_train = encoder.transform(df_train['labels_train'])
    
    encoder.fit(df_val['labels_val'])
    labels_val = encoder.transform(df_val['labels_val'])

    encoder.fit(df_test['labels_test'])
    labels_test = encoder.transform(df_test['labels_test'])
    
    
    #Encode amino acid sequences
    aa_seqs_len = extract_upstream_aa + 1 + extract_downstream_aa
    
    tokenizer_aa = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", 
                                          do_lower_case=False, 
                                          model_max_length=aa_seqs_len + 2)

    train_encodings_aa = tokenizer_aa(list(df_train['aa_sequences_train']), 
                                padding=True,  #pad sequences to max length and apply attention mask
                                truncation=True, 
                                return_tensors="pt")
    attention_mask_train = train_encodings_aa['input_ids'] != tokenizer_aa.pad_token_id
    train_encodings_aa['attention_mask'] = attention_mask_train.int()  # Replace the old attention mask

    val_encodings_aa = tokenizer_aa(list(df_val['aa_sequences_val']), 
                              padding=True,  #pad sequences to max length and apply attention mask
                              truncation=True,
                              return_tensors="pt")
    attention_mask_val = val_encodings_aa['input_ids'] != tokenizer_aa.pad_token_id
    val_encodings_aa['attention_mask'] = attention_mask_val.int()  # Replace the old attention mask

    test_encodings_aa = tokenizer_aa(list(df_test['aa_sequences_test']), 
                               padding=True,  #pad sequences to max length and apply attention mask
                               truncation=True,
                               return_tensors="pt")
    attention_mask_test = test_encodings_aa['input_ids'] != tokenizer_aa.pad_token_id
    test_encodings_aa['attention_mask'] = attention_mask_test.int()  # Replace the old attention mask
    
    return labels_train, train_encodings_aa, \
           labels_val, val_encodings_aa, \
           labels_test, test_encodings_aa


class MultiInputDataset(torch.utils.data.Dataset):
    """
    This class is used to create a dataset for inputting to the model with multiple types of inputs.
    """
    def __init__(self, aa_encodings, group, labels):
        self.aa_encodings = aa_encodings
        self.group = group
        self.labels = labels

    def __getitem__(self, idx):
        
        item = {
            'aa_encodings': {key: torch.as_tensor(val[idx]) for key, val in self.aa_encodings.items()},
            'group': self.group[idx],
            'labels': torch.as_tensor(self.labels[idx])
        }
        return item

    def __len__(self):
        return len(self.labels)


class ESM2Model(nn.Module):
    """
    Define model architecture.
    """
    def __init__(self, 
                 aa_encoding_length, 
                 num_labels = 1):
        
        super(ESM2Model, self).__init__()

        ###Define layers###
        self.pretrained_model_aa = AutoModel.from_pretrained("../../data/data_model/pretrained_models/finetuned_models/"+esm2_pretrained)
    
        #Dont update weights on finetuned models
        for param in self.pretrained_model_aa.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(self.pretrained_model_aa.config.hidden_size*aa_encoding_length, num_labels)

        #Define binary output
        self.num_labels = num_labels

    def forward(self, 
                x_aa, 
                attention_mask_aa):
        
        features_aa = self.pretrained_model_aa(x_aa, attention_mask = attention_mask_aa)
        
        #Get entire sequence representation from last hidden state from both pretrained LMs
        sequence_output_aa = features_aa['last_hidden_state']

        #Reshape outputs to torch-size([minibatch_size, sequence_length*hidden_size])
        reshaped_output_aa = sequence_output_aa.view(sequence_output_aa.size(0), -1)

        output = self.classifier(reshaped_output_aa)
        
        return output


###Main code###
#Load Data
data_train, data_val, data_test = load_and_partition_data(model_no = model_no)

#Get data
df_train, df_val, df_test = extract_datasets(data_train, 
                                             data_val, 
                                             data_test, 
                                             extract_upstream_aa = aa_upstream,  
                                             extract_downstream_aa = aa_downstream)

#Get sequence and label encodings 
labels_train, train_encodings_aa,  \
labels_val, val_encodings_aa,  \
labels_test, test_encodings_aa = create_encodings(df_train, df_val, df_test, extract_upstream_aa = aa_upstream, extract_downstream_aa = aa_downstream)

aa_encodings_len = train_encodings_aa['input_ids'].shape[1]

#Get all datatypes for model in one dataset
train_dataset = MultiInputDataset(train_encodings_aa, df_train["group_train"], labels_train)
val_dataset = MultiInputDataset(val_encodings_aa, df_val["group_val"], labels_val)
test_dataset = MultiInputDataset(test_encodings_aa, df_test["group_test"], labels_test)

#Define data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=8,  # Adjust based on your CPU cores
    pin_memory=True # Optimizes data transfer to GPU
    )

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=8, 
    pin_memory=True)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=8,  
    pin_memory=True)


#Instantiate model
esm2_model = ESM2Model(aa_encoding_length = aa_encodings_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {device}")
esm2_model.to(device)

pos_weight = torch.tensor(3, dtype=torch.float).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#Define training settings
num_epochs = 5
optimizer = optim.Adam(esm2_model.parameters(), lr=lr)

#Define how often to evaluate
steps_per_epoch = len(train_dataset) / (batch_size)                             #The number of steps taken per epoch
eval_per_epoch = 16                                                             #The number of evaluations made per epoch
steps_between_evals = int(steps_per_epoch / eval_per_epoch)                     #The number of steps to take between each evaluation
total_eval_steps = eval_per_epoch * num_epochs                                  #The total number of times the model is evaluated during finetuning

print(f"Running for {num_epochs} epochs with a batchsize of {batch_size}. Total number of training sequences: {len(train_dataset)}.", sep = "", flush=True)
print(f"Evaluates every {steps_between_evals} steps.", sep = "", flush=True)


#Loss across epochs
loss_train_steps = []
loss_val_steps = []
step_number = []
epoch_number = []

#Initialize variables for early stopping
best_val_loss = float('inf')
threshold_patience = 12  #Number of evaluations with no improvement to wait before stopping
counter_patience = 0

train_loss = 0.0
step = 0

wandb.init(
    #set the wandb project where this run will be logged
    project="train_netstart2_ablation",

    config = {
        "model_number": str(model_no)
    }
)


print("Initializing training", flush = True)
#Training loop
for epoch in range(num_epochs):
    print("Epoch: ", epoch+1, flush=True)

    for batch in train_loader:
        esm2_model.train()
        #Get inputs and labels
        inputs_aa = batch["aa_encodings"]["input_ids"].to(device)
        attention_mask_aa = batch["aa_encodings"]["attention_mask"].to(device)
        labels = batch['labels'].to(device).view(-1, 1).float()

        torch.cuda.empty_cache() 

        #Forward pass
        outputs = esm2_model(inputs_aa, attention_mask_aa)
        loss = loss_fn(outputs, labels)

        #Backward pass
        optimizer.zero_grad()
        loss.backward()

        #Update parameters
        optimizer.step()

        train_loss += loss.item()

        torch.cuda.empty_cache()
        del outputs
        del loss
        
        step += 1

        if step % int(steps_between_evals/4) == 0:
            print(f"Step {step} / {int(steps_per_epoch)*5}", sep = " ", flush=True)

        # Make evaluation of model
        if step % steps_between_evals == 0:
            esm2_model.eval()

            val_loss = 0.0
            val_true_labels = []
            val_predicted_probs = []

            # Group-wise dictionaries
            groupwise_true_labels = defaultdict(list)
            groupwise_predicted_probs = defaultdict(list)
            groupwise_losses = defaultdict(float)

            with torch.no_grad():
                for count, batch in enumerate(val_loader):
                    # Get inputs and labels
                    inputs_aa = batch["aa_encodings"]["input_ids"].to(device)
                    attention_mask_aa = batch["aa_encodings"]["attention_mask"].to(device)
                    group = batch["group"]  # Organism group
                    labels = batch['labels'].to(device).view(-1, 1).float()

                    # Forward pass
                    outputs = esm2_model(inputs_aa, attention_mask_aa)
                    loss = loss_fn(outputs, labels)

                    val_loss += loss.item()

                    # Aggregate overall metrics
                    val_true_labels.extend(labels.cpu().numpy().flatten())
                    val_predicted_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())

                    # Aggregate metrics by group
                    for i, g in enumerate(group):
                        groupwise_true_labels[g].append(labels[i].item())
                        groupwise_predicted_probs[g].append(torch.sigmoid(outputs[i]).item())
                        groupwise_losses[g] += loss.item()

                # Calculate overall metrics
                train_avg_loss = round(train_loss / steps_between_evals, 6)
                val_avg_loss = round(val_loss / len(val_loader), 6)
                val_auc_roc = round(roc_auc_score(val_true_labels, val_predicted_probs), 6)
                precision, recall, _ = precision_recall_curve(val_true_labels, val_predicted_probs)
                val_avg_precision = average_precision_score(val_true_labels, val_predicted_probs)
                val_pr_auc = round(auc(recall, precision), 6)

                del loss
                del outputs
                torch.cuda.empty_cache()

                # Log overall metrics
                wandb.log({"train_loss": train_avg_loss, "val_loss": val_avg_loss, 
                           "Val AUROC": val_auc_roc, "Val AUPR": val_pr_auc, 
                           "Val Precision": val_avg_precision})

                # Log group-specific metrics
                for g in groupwise_true_labels.keys():
                    group_true = groupwise_true_labels[g]
                    group_preds = groupwise_predicted_probs[g]
                    group_loss = groupwise_losses[g] / len(group_true)

                    # Compute group-specific AUROC
                    group_auroc = round(roc_auc_score(group_true, group_preds), 4)

                    # Log metrics for the group
                    wandb.log({
                        f"{g} val loss": group_loss,
                        f"{g} val AUROC": group_auroc,
                    })

                # Store loss progression
                loss_train_steps.append(train_avg_loss)
                loss_val_steps.append(val_avg_loss)
                step_number.append(step)
                epoch_number.append(epoch + 1)

                # Re-initialize
                train_loss = 0.0

                print(f"Train Loss: {train_avg_loss}\tVal Loss: {val_avg_loss}\t"
                      f"Val AUC ROC: {val_auc_roc}\tVal AUC PR: {val_pr_auc}",
                      sep="", flush=True)

                # Check validation loss, apply early stopping if necessary
                if val_avg_loss < best_val_loss:
                    # Update lowest loss
                    best_val_loss = val_avg_loss

                    # Reset counter
                    counter_patience = 0

                    # Get the state dict (strip 'module.' if needed)
                    if isinstance(esm2_model, torch.nn.DataParallel) or isinstance(esm2_model, torch.nn.parallel.DistributedDataParallel):
                        state_dict = esm2_model.module.state_dict()  # Strips 'module.' prefix
                    else:
                        state_dict = esm2_model.state_dict()

                    # Save the model parameters
                    torch.save(state_dict, '../../data/data_model_ablations/models/' + esm2_model_filename + ".pth")
                else:
                    counter_patience += 1

                # Early stopping condition
                if counter_patience >= threshold_patience:
                    print("Early stopping. No improvement in validation loss.", flush=True)
                    break

    #When the model is overfitting, use early stopping
    if counter_patience >= threshold_patience:
        break

wandb.finish()

#Test Performance measurement
test_true_labels = []  # Store true labels for ROC AUC calculation
test_predicted_probs = []  # Store predicted probabilities for ROC AUC calculation

with torch.no_grad():
    for batch in test_loader:
        inputs_aa = batch["aa_encodings"]["input_ids"].to(device)
        attention_mask_aa = batch["aa_encodings"]["attention_mask"].to(device)
        labels = batch['labels'].to(device).view(-1, 1).float()

        #Get predctions and loss
        outputs = esm2_model(inputs_aa, attention_mask_aa)

        # Append true labels and predicted probabilities
        test_true_labels.extend(labels.cpu().numpy().flatten())
        test_predicted_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())

        
##Measure performance at different MCC thresholds
thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
mccs = []

for threshold in thresholds:
    mccs.append(matthews_corrcoef(np.array(test_true_labels), (np.array(test_predicted_probs) >= threshold).astype(int)))

best_threshold, best_mcc = max(zip(thresholds, mccs), key=lambda x: x[1])

print("Best threshold:", best_threshold, flush = True)


#Apply the best threshold to get the predicted labels
predicted = (np.array(test_predicted_probs) > best_threshold).astype(float)

correct = (predicted == test_true_labels).sum().item()

total = len(test_true_labels)
test_accuracy = correct/total

#Calculate ROC AUC score
test_roc_auc = roc_auc_score(test_true_labels, test_predicted_probs)

#Calculate PR AUC score
precision, recall, _ = precision_recall_curve(test_true_labels, test_predicted_probs)
test_pr_auc = auc(recall, precision)

TP = sum((test_true_labels == 1.0) and (test_predicted_probs >= best_threshold) for test_true_labels, test_predicted_probs in zip(test_true_labels, test_predicted_probs))
FN = sum((test_true_labels == 1.0) and (test_predicted_probs < best_threshold) for test_true_labels, test_predicted_probs in zip(test_true_labels, test_predicted_probs))
sensitivity = TP / (TP + FN)

TN = sum((test_true_labels == 0.0) and (test_predicted_probs < best_threshold) for test_true_labels, test_predicted_probs in zip(test_true_labels, test_predicted_probs))
FP = sum((test_true_labels == 0.0) and (test_predicted_probs >= best_threshold) for test_true_labels, test_predicted_probs in zip(test_true_labels, test_predicted_probs))
specificity = TN / (TN + FP)

#Calculate Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(np.array(test_true_labels), (np.array(test_predicted_probs) >= best_threshold).astype(int))

assert total == TP + FN + TN + FP, "Something went wrong calculating test performance"

print("Test Metrics:", flush = True)
print(f"Test Accuracy: {test_accuracy:.4f}", flush = True)
print(f"Test ROC AUC: {test_roc_auc:.4f}, \nTest PR AUC: {test_pr_auc:.4f}", flush = True)
print(f"Test Sensitivity: {sensitivity:.4f}, \nTest Specificity: {specificity:.4f}\nTest MCC: {mcc:.4f}", flush = True)