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
from sklearn.metrics import average_precision_score

import wandb

torch.cuda.is_available()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

#train all 4 models with the optimized, respective hyperparameters
model_no = 1
hyperpars_filename = 'netstart1_model'+str(model_no)+'_hyperparameters.json'
netstart1_model_filename = "netstart1_model" + str(model_no) + "_ablation"

#Collect optimized hyperparameters
with open("../../data/data_model_ablations/hyperparameters/"+hyperpars_filename, 'r') as json_file:
    model_config = json.load(json_file)

#Define hyperparameters
nt_upstream = model_config["nt_upstream"]
nt_downstream = model_config["nt_downstream"]
batch_size = model_config["batch_size"]
emb_size_tax = model_config["emb_size_tax"]
hidden_neurons_nt = model_config["hidden_neurons_nt"]
hidden_neurons_merge = model_config["hidden_neurons_merge"]
dropout_rate_1 = model_config["dropout_rate_1"]
dropout_rate_2 = model_config["dropout_rate_2"]
depth_nt_window = model_config["depth_nt_window"]
lr = model_config["lr"]

def set_seed(seed):
    """
    Set seed for reproducibility

    Args:
        seed (int): seed value to set
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_partition_data(model_no):
    """
    Load in data and create train and validation splits

    Args:
        model_no (int [1, 2, 3, 4]): We are training 4 models; model no. corresponds to defining
                                     which partitions are train and which is validation
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

    #Extract datasplits for train and valiation sets for the different models to train
    if model_no == 1:
        #Concat train partitions
        data_train = pd.concat([data_partition_2,
                                data_partition_3,
                                data_partition_4],
                               ignore_index = True)
        data_val = data_partition_1

    elif model_no == 2:
        data_train = pd.concat([data_partition_1,
                                data_partition_3,
                                data_partition_4],
                               ignore_index = True)
        data_val = data_partition_2

    if model_no == 3:
        data_train = pd.concat([data_partition_1,
                                data_partition_2,
                                data_partition_4],
                               ignore_index = True)
        data_val = data_partition_3

    if model_no == 4:
        data_train = pd.concat([data_partition_1,
                                data_partition_2,
                                data_partition_3],
                               ignore_index = True)
        data_val = data_partition_4

    #Replace 'Canis lupus familiaris' with 'Canis lupus'
    data_train['Species'] = data_train['Species'].replace('Canis lupus familiaris', 'Canis lupus')
    data_val['Species'] = data_val['Species'].replace('Canis lupus familiaris', 'Canis lupus')


    return data_train, data_val


def load_taxonomy_mappings():
    """
    Load major taxonomy ranks (species, genus, family, order, class, phylum, kingdom)
    for each species in dataset.
    """

    #Read in json files with taxonomy information
    with open("../../data/data_model/taxonomy/species_names.json", 'r') as file:
        species_names_dict = json.load(file)
    with open("../../data/data_model/taxonomy/species_to_kingdom.json", 'r') as file:
        species_kingdom_dict = json.load(file)
    with open("../../data/data_model/taxonomy/species_to_phylum.json", 'r') as file:
        species_phylum_dict = json.load(file)
    with open("../../data/data_model/taxonomy/species_to_class.json", 'r') as file:
        species_class_dict = json.load(file)
    with open("../../data/data_model/taxonomy/species_to_order.json", 'r') as file:
        species_order_dict = json.load(file)
    with open("../../data/data_model/taxonomy/species_to_family.json", 'r') as file:
        species_family_dict = json.load(file)
    with open("../../data/data_model/taxonomy/species_to_genus.json", 'r') as file:
        species_genus_dict = json.load(file)

    #Create a list of unique IDs from all dictionaries, excluding '0'
    #(missing value / species not classified at given rank)
    all_ids = set(species_names_dict.values()) | \
              set(species_genus_dict.keys()) | \
              set(species_family_dict.keys()) | \
              set(species_order_dict.keys()) | \
              set(species_class_dict.keys()) | \
              set(species_phylum_dict.keys()) | \
              set(species_kingdom_dict.keys())

    all_ids.discard('0')

    #Generate mappings for all taxonomic ranks except '0'
    species_id_mapping = {ncbi_id: str(i + 1) for i, ncbi_id in enumerate(sorted(all_ids))}

    taxonomic_dicts = [species_genus_dict, species_family_dict, species_order_dict,
                       species_class_dict, species_phylum_dict, species_kingdom_dict]

    taxonomic_id_mappings = [{ncbi_id: str(i + 1) for i, ncbi_id in enumerate(sorted(set(d.values())))} for d in taxonomic_dicts]

    #Create the final output mapping to feed into model
    tax_mapping = {}
    for species_name, species_id in species_names_dict.items():
        taxonomic_ids = []
        for idx, taxonomic_dict in enumerate(taxonomic_dicts):
            taxonomic_id = taxonomic_dict.get(species_id, "")
            if taxonomic_id != '0':
                taxonomic_id = taxonomic_id_mappings[idx].get(taxonomic_id, "")
            taxonomic_ids.append(taxonomic_id)

        species_id = species_id_mapping.get(species_id, "")
        tax_mapping[species_name] = [species_id] + taxonomic_ids

    #Initialize a list with seven zeros ([Species id, Genus id, ..., Kingom id])
    vocab_sizes = [0] * 7

    for values in tax_mapping.values():
        for idx, val in enumerate(values):
            vocab_sizes[idx] = max(vocab_sizes[idx], int(val) + 1)

    #vocab_sizes: the number of different taxonomic groups present in each rank [species, genus, ... kingdom]
    #tax_mapping: taxonomical mapping of each organism based on each rank
    return vocab_sizes, tax_mapping


def extract_datasets(data_train,
                     data_val,
                     tax_mapping,
                     extract_upstream_nt,
                     extract_downstream_nt):
    """
    Create the datasets to use for model development.

    Args:
        data_train (df): the loaded dataframe with train partitions
        data_val (df): the loaded dataframe with validation partition
        tax_mapping (dict): dictionary with mappings of taxonomic ranks for each species
        extract_upstream_nt (int): the number of nucleotide positions upstream ATG to extract
        extract_downstream_nt (int): the number of nucleotide positions downstream ATG to extract
    """

    #Sequences in dataset consists of 603 nucleotides, labelled ATG placed on position 300
    ATG_position = 300
    
    #Get all training data
    rows = []
    for i, row in enumerate(data_train.itertuples()):
        taxonomy_levels = tax_mapping[row.Species]

        #Append each row of wanted information to the list
        rows.append({
            'nt_sequences_train': row.Sequence[ATG_position-extract_upstream_nt:ATG_position+3+extract_downstream_nt],
            'tax_ranks_train': np.array(taxonomy_levels),
            'labels_train': int(row.TIS)
        })
    
    #Store data in df
    df_train = pd.DataFrame(rows)
    
    #Get all validation data
    rows = []
    for i, row in enumerate(data_val.itertuples()):
        taxonomy_levels = tax_mapping[row.Species] 
        
        rows.append({
            'nt_sequences_val': row.Sequence[ATG_position-extract_upstream_nt:ATG_position+3+extract_downstream_nt],
            'tax_ranks_val': np.array(taxonomy_levels),
            'labels_val': int(row.TIS)
        })
        
    df_val = pd.DataFrame(rows)


    return df_train, df_val


def one_hot_encode(sequence):
    """
    One hot encode nucleotide sequences; get matrix format of 4 rows, len(sequence) columns. 
    
    Args:
        sequence (str): nucleotide sequence
    """
    
    nucleotides = ['A', 'C', 'G', 'T', 'N']
    encoding = torch.zeros(len(nucleotides) - 1, len(sequence))  # Exclude 'N'
    
    for i, char in enumerate(sequence):
        if char in nucleotides[:4]:  # Only encode A, C, G, T
            encoding[nucleotides.index(char)][i] = 1
        # For 'N', the corresponding column remains all zeros
    
    return encoding

def create_encodings(df_train, df_val):
    """
    Encode labels and sequences to fit input to pretrained models.

    Args:
        df_train (dataframe): dataframe with training data
        df_val (dataframe): dataframe with validation data
    """

    #Encode labels to ensure correct format
    encoder = LabelEncoder()
    encoder.fit(df_train['labels_train'])
    labels_train = encoder.transform(df_train['labels_train'])

    encoder.fit(df_val['labels_val'])
    labels_val = encoder.transform(df_val['labels_val'])

    #One-hot encode nucleotide sequences (local start codon context) and format them
    sequences_train = [one_hot_encode(seq) for seq in df_train['nt_sequences_train']]
    sequences_train_nt = torch.stack(sequences_train)
    sequences_val = [one_hot_encode(seq) for seq in df_val['nt_sequences_val']]
    sequences_val_nt = torch.stack(sequences_val)

    return labels_train, sequences_train_nt, \
           labels_val, sequences_val_nt


class MultiInputDataset(torch.utils.data.Dataset):
    """
    This class is used to create a dataset for inputting to the model with multiple types of inputs.
    """
    def __init__(self, nt_encodings, tax_ranks, labels):
        self.nt_encodings = nt_encodings
        self.tax_ranks = tax_ranks
        self.labels = labels

    def __getitem__(self, idx):
        #Convert strings to integers in the 'tax_ranks' list
        tax_ranks_item = [int(rank) if rank != '0' else 0 for rank in self.tax_ranks[idx]]

        #convert to tensors
        item = {
            'nt_encodings': torch.as_tensor(self.nt_encodings[idx], dtype=torch.float32),				#get one-hot encoded sequences
            'tax_ranks': torch.LongTensor(tax_ranks_item),												#get taxonomical ranks
            'labels': torch.as_tensor(self.labels[idx])													#get labels
        }
        return item

    def __len__(self):
        return len(self.labels)

class NetStart1Model(nn.Module):
    """
    Define model architecture of NetStart 2.0
    """
    def __init__(self,
                 nt_encoding_length,
                 vocab_sizes,
                 emb_size = emb_size_tax,
                 hidden_neurons_nt = hidden_neurons_nt,
                 hidden_neurons_merge = hidden_neurons_merge,
                 dropout_rate_1 = dropout_rate_1,
                 dropout_rate_2 = dropout_rate_2,
                 num_hidden_layers_nt = depth_nt_window,
                 num_labels=1):

        super(NetStart1Model, self).__init__()

        # Embedding size per taxonomical rank represented
        self.emb_size = emb_size

        ### Define layers ###

        # Define feedforward hidden layers for local start codon context window
        self.nt_layers = nn.ModuleList()
        self.nt_layers.append(nn.Linear(nt_encoding_length, hidden_neurons_nt))  # First layer

        # Additional hidden layers for nt encoding if `num_hidden_layers_nt > 1`
        for _ in range(num_hidden_layers_nt - 1):
            self.nt_layers.append(nn.Linear(hidden_neurons_nt, hidden_neurons_nt))

        # Define taxonomical embeddings
        self.embedding_species = nn.Embedding(vocab_sizes[0], emb_size, padding_idx=0)
        self.embedding_genus = nn.Embedding(vocab_sizes[1], emb_size, padding_idx=0)
        self.embedding_family = nn.Embedding(vocab_sizes[2], emb_size, padding_idx=0)
        self.embedding_order = nn.Embedding(vocab_sizes[3], emb_size, padding_idx=0)
        self.embedding_class = nn.Embedding(vocab_sizes[4], emb_size, padding_idx=0)
        self.embedding_phylum = nn.Embedding(vocab_sizes[5], emb_size, padding_idx=0)
        self.embedding_kingdom = nn.Embedding(vocab_sizes[6], emb_size, padding_idx=0)

        # Define shared feedforward layer
        self.hidden_layer_2 = nn.Linear(hidden_neurons_nt + emb_size, hidden_neurons_merge)

        # Define classifier
        self.classifier = nn.Linear(hidden_neurons_merge, num_labels)

        # Define dropout rates
        self.dropout_1 = nn.Dropout(dropout_rate_1)
        self.dropout_2 = nn.Dropout(dropout_rate_2)

        # Define binary output
        self.num_labels = num_labels

        # Activation Functions
        self.relu = nn.ReLU()

    def forward(self,
                x_nt,
                tax_ranks):

        # Reshape the nucleotide input to flatten it (4xnucleotides to 1D tensor)
        x_nt = x_nt.view(x_nt.size(0), -1)

        # Define hidden layer for local start codon context sequence representations
        hidden_nt = x_nt
        for layer in self.nt_layers:
            hidden_nt = layer(hidden_nt)
            hidden_nt = self.relu(hidden_nt)
            hidden_nt = self.dropout_1(hidden_nt)

        # Define embeddings for the major taxonomical ranks
        species_embedding = self.embedding_species(tax_ranks[:, 0]).unsqueeze(1)
        genus_embedding = self.embedding_genus(tax_ranks[:, 1]).unsqueeze(1)
        family_embedding = self.embedding_family(tax_ranks[:, 2]).unsqueeze(1)
        order_embedding = self.embedding_order(tax_ranks[:, 3]).unsqueeze(1)
        class_embedding = self.embedding_class(tax_ranks[:, 4]).unsqueeze(1)
        phylum_embedding = self.embedding_phylum(tax_ranks[:, 5]).unsqueeze(1)
        kingdom_embedding = self.embedding_kingdom(tax_ranks[:, 6]).unsqueeze(1)

        embeddings_list = [
            species_embedding,
            genus_embedding,
            family_embedding,
            order_embedding,
            class_embedding,
            phylum_embedding,
            kingdom_embedding
        ]

        # Sum embeddings from each rank
        stacked_embeddings = torch.stack(embeddings_list, axis=1).sum(axis=1)
        embeddings_tax = stacked_embeddings.view(stacked_embeddings.size(0), -1)

        # Concatenate all inputs
        concatenated_input = torch.cat([hidden_nt, embeddings_tax], dim=1)

        # Define hidden layer to feed all input types through together
        hidden = self.hidden_layer_2(concatenated_input)
        hidden = self.relu(hidden)
        hidden = self.dropout_2(hidden)

        # Define classification layer
        output = self.classifier(hidden)

        return output
    

###Main code###
#Load Data
data_train, data_val = load_and_partition_data(model_no = model_no)

#Load Taxonomy Mappings
vocab_sizes, tax_mapping = load_taxonomy_mappings()

#Get data
df_train, df_val = extract_datasets(data_train, 
                                    data_val,
                                    tax_mapping,
                                    extract_upstream_nt = nt_upstream, 
                                    extract_downstream_nt = nt_downstream)

val_samples_size = df_val.shape[0]

#Get sequence and label encodings 
labels_train, train_encodings_nt, \
labels_val, val_encodings_nt = create_encodings(df_train, 
                                                                     df_val)

nt_encodings_len_flattened = len(train_encodings_nt[0])*len(train_encodings_nt[0][0]) #4 * numbers of nt

#Get all datatypes for model in one dataset
train_dataset = MultiInputDataset(train_encodings_nt, df_train['tax_ranks_train'],  labels_train)
val_dataset = MultiInputDataset(val_encodings_nt, df_val['tax_ranks_val'], labels_val)

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


#Instantiate model
netstart1_model = NetStart1Model(nt_encoding_length = nt_encodings_len_flattened, 
                               vocab_sizes = vocab_sizes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {device}")
netstart1_model.to(device)

pos_weight = torch.tensor(3, dtype=torch.float).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


#Define training settings
num_epochs = 5
optimizer = optim.Adam(netstart1_model.parameters(), lr=lr)

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
    project="train_netstart1",

    config = {
        "model_number": str(model_no)
    }
)

print("Initializing training", flush = True)
#Training loop
for epoch in range(num_epochs):
    print("Epoch: ", epoch+1, flush=True)

    for batch in train_loader:
        netstart1_model.train()
        #Get inputs and labels
        inputs_nt = batch["nt_encodings"].to(device)
        tax_ranks = batch["tax_ranks"].to(device)
        labels = batch['labels'].to(device).view(-1, 1).float()

        torch.cuda.empty_cache() 

        #Forward pass
        outputs = netstart1_model(inputs_nt, tax_ranks)
        loss = loss_fn(outputs, labels)

        #Backward pass
        optimizer.zero_grad()
        loss.backward()

        #Update parameters
        optimizer.step()

        train_loss += loss.item()

        del outputs
        del loss
        
        step += 1

        if step % int(steps_between_evals/4) == 0:
            print(f"Step {step} / {int(steps_per_epoch)*5}", sep = " ", flush=True)

        # Make evaluation of model
        if step % steps_between_evals == 0:
            netstart1_model.eval()

            val_loss = 0.0
            val_true_labels = []
            val_predicted_probs = []

            with torch.no_grad():
                for batch in val_loader:
                    # Get inputs and labels
                    inputs_nt = batch["nt_encodings"].to(device)
                    tax_ranks = batch["tax_ranks"].to(device)
                    labels = batch['labels'].to(device).view(-1, 1).float()

                    # Forward pass
                    outputs = netstart1_model(inputs_nt, tax_ranks)
                    loss = loss_fn(outputs, labels)

                    val_loss += loss.item()

                    # Aggregate overall metrics
                    val_true_labels.extend(labels.cpu().numpy().flatten())
                    val_predicted_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())

                    # Check for NaNs in predicted probabilities
                    if np.any(np.isnan(labels.cpu().numpy().flatten())):
                        print("NaN detected in val_true_labels", flush = True)
                        print(labels.cpu().numpy().flatten(), flush = True)

                    # Check for NaNs in true labels (unlikely but good to check)
                    if np.any(np.isnan(torch.sigmoid(outputs).cpu().numpy().flatten())):
                        print("NaN detected in val_predicted_probs:", flush = True)
                        print(torch.sigmoid(outputs).cpu().numpy().flatten(), flush = True)
                        
                print("Val True Labels len: ", len(val_true_labels))
                print("Val predicted probs len: ", len(val_predicted_probs))

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
                    if isinstance(netstart1_model, torch.nn.DataParallel) or isinstance(netstart1_model, torch.nn.parallel.DistributedDataParallel):
                        state_dict = netstart1_model.module.state_dict()  # Strips 'module.' prefix
                    else:
                        state_dict = netstart1_model.state_dict()

                    # Save the model parameters
                    torch.save(state_dict, '../../data/data_model_ablations/models/' + netstart1_model_filename + ".pth")
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

print("Training finished.")