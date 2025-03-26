import os
import numpy as np
import pandas as pd
import json
import wandb
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from transformers import AutoTokenizer, AutoModel

#Clear the GPU memory cache
torch.cuda.empty_cache()

#Configure CUDA memory allocations (helps manage fragmentation in the GPU memory)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

#Run hyperparameter tuning for model_no 1, 2, 3, 4
model_no = 1

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
    Load in data and create train, validation and test splits

    Args:
        model_no (int [1, 2, 3, 4]): We are training 4 models; model no. corresponds to defining
                                     which partitions are train and which is validation
    
    Returns:
        data_train (df): dataframe with training data
        data_val (df): dataframe with validation data
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

    elif model_no == 3:
        data_train = pd.concat([data_partition_1,
                                data_partition_2,
                                data_partition_4],
                               ignore_index = True)
        data_val = data_partition_3

    elif model_no == 4:
        data_train = pd.concat([data_partition_1,
                                data_partition_2,
                                data_partition_3],
                               ignore_index = True)
        data_val = data_partition_4

    #Replace 'Canis lupus familiaris' with 'Canis lupus'
    data_train['Species'] = data_train['Species'].replace('Canis lupus familiaris', 'Canis lupus')
    data_val['Species'] = data_val['Species'].replace('Canis lupus familiaris', 'Canis lupus')

    return data_train, data_val

def encode_nucleotide_to_amino_acid(sequence):
    """
    The function takes a nucleotide sequence and translates it into the
    corresponding amino acid sequence

    Args:
        sequence (str): A nucleotide sequence

    Returns:      
        amino_acid_sequence (str): The amino acid sequence
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


def load_taxonomy_mappings():
    """
    Load major taxonomy ranks (species, genus, family, order, class, phylum, kingdom)
    for each species in dataset.

    Returns:
        vocab_sizes (list): the number of different taxonomic groups present in each rank [species, genus, ... kingdom]
        tax_mapping (dict): taxonomical mapping of each organism based on each rank
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
                     extract_downstream_nt,
                     extract_upstream_aa,
                     extract_downstream_aa):
    """
    Create the datasets to use for model development.

    Args:
        data_train (df): the loaded dataframe with train partitions
        data_val (df): the loaded dataframe with validation partition
        tax_mapping (dict): dictionary with mappings of taxonomic ranks for each species
        extract_upstream_nt (int): the number of nucleotide positions upstream ATG to extract
        extract_downstream_nt (int): the number of nucleotide positions downstream ATG to extract
        extract_upstream_aa (int): the number of amino acid positions upstream ATG to extract
        extract_downstream_aa (int): the number of amino acid positions downstream ATG to extract
    
    Returns:
        df_train (df): dataframe with training data
        df_val (df): dataframe with validation data
    """

    #Sequences in dataset consists of 603 nucleotides, labelled ATG placed on position 300
    ATG_position = 300
    
    #Get all training data
    rows = []
    for i, row in enumerate(data_train.itertuples()):
        taxonomy_levels = tax_mapping[row.Species]

        #Append each row of wanted information to the list
        rows.append({
            'aa_sequences_train': encode_nucleotide_to_amino_acid(row.Sequence[ATG_position-3*extract_upstream_aa:ATG_position+3+3*extract_downstream_aa]),
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
            'aa_sequences_val': encode_nucleotide_to_amino_acid(row.Sequence[ATG_position-3*extract_upstream_aa:ATG_position+3+3*extract_downstream_aa]),
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

    Returns: 
        encoding (tensor): one-hot encoded tensor of the sequence
    """
    
    nucleotides = ['A', 'C', 'G', 'T', 'N']
    encoding = torch.zeros(len(nucleotides) - 1, len(sequence))  # Exclude 'N'
    
    for i, char in enumerate(sequence):
        if char in nucleotides[:4]:  # Only encode A, C, G, T
            encoding[nucleotides.index(char)][i] = 1
        # For 'N', the corresponding column remains all zeros
    
    return encoding

def create_encodings(df_train, df_val, extract_upstream_aa, extract_downstream_aa):
    """
    Encode labels and sequences to fit input to pretrained models.

    Args:
        df_train (dataframe): dataframe with training data
        df_val (dataframe): dataframe with validation data
        extract_upstream_aa (int): number of amino acids upstream TIS to extract
        extract_downstream_aa (int): number of amino acids downstream TIS to extract
    
    Returns:
        labels_train (array): encoded labels for training data
        train_encodings_aa (dict): encoded amino acid sequences for training data
        sequences_train_nt (tensor): one-hot encoded nucleotide sequences for training data
        labels_val (array): encoded labels for validation data
        val_encodings_aa (dict): encoded amino acid sequences for validation data
        sequences_val_nt (tensor): one-hot encoded nucleotide sequences for validation data
    """

    #Encode labels to ensure correct format
    encoder = LabelEncoder()
    encoder.fit(df_train['labels_train'])
    labels_train = encoder.transform(df_train['labels_train'])

    encoder.fit(df_val['labels_val'])
    labels_val = encoder.transform(df_val['labels_val'])


    #Get amino acid sequence length
    aa_seqs_len = extract_upstream_aa + 1 + extract_downstream_aa

    #Load ESM2 tokenizer
    tokenizer_aa = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D",
                                          do_lower_case=False,
                                          model_max_length=aa_seqs_len + 2)

    #Encode amino acid sequences
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

    #One-hot encode nucleotide sequences (local start codon context) and format them
    sequences_train = [one_hot_encode(seq) for seq in df_train['nt_sequences_train']]
    sequences_train_nt = torch.stack(sequences_train)
    sequences_val = [one_hot_encode(seq) for seq in df_val['nt_sequences_val']]
    sequences_val_nt = torch.stack(sequences_val)

    return labels_train, train_encodings_aa, sequences_train_nt, \
           labels_val, val_encodings_aa, sequences_val_nt


class MultiInputDataset(torch.utils.data.Dataset):
    """
    This class is used to create a dataset for inputting to the model with multiple types of inputs.

    Args:  
        aa_encodings (dict): dictionary with amino acid encodings
        nt_encodings (tensor): one-hot encoded nucleotide sequences
        tax_ranks (list): taxonomical ranks
        labels (array): encoded labels
    
    Returns:
        item (dict): dictionary with all inputs and labels
    """
    def __init__(self, aa_encodings, nt_encodings, tax_ranks, labels):
                 self.aa_encodings = aa_encodings
                 self.nt_encodings = nt_encodings
                 self.tax_ranks = tax_ranks
                 self.labels = labels

    def __getitem__(self, idx):
        #Convert strings to integers in the 'tax_ranks' list
        tax_ranks_item = [int(rank) if rank != '0' else 0 for rank in self.tax_ranks[idx]]

        #convert to tensors
        item = {
            'aa_encodings': {key: torch.as_tensor(val[idx]) for key, val in self.aa_encodings.items()}, #get both inputs and attention masks
            'nt_encodings': torch.as_tensor(self.nt_encodings[idx], dtype=torch.float32),				#get one-hot encoded sequences
            'tax_ranks': torch.LongTensor(tax_ranks_item),												#get taxonomical ranks
            'labels': torch.as_tensor(self.labels[idx])													#get labels
        }
        return item

    def __len__(self):
        return len(self.labels)


class NetstartModel(nn.Module):
    """
    Define model architecture of NetStart 2.0

    Args:
        model_no (int): model number
        aa_encoding_length (int): length of amino acid encoding
        nt_encoding_length (int): length of nucleotide encoding
        vocab_sizes (list): list of vocabulary sizes for each taxonomical rank
        emb_size (int): embedding size
        hidden_neurons_aa (int): number of neurons in hidden layer for amino acid encoding
        hidden_neurons_nt (int): number of neurons in hidden layer for nucleotide encoding
        hidden_neurons_merge (int): number of neurons in hidden layer for merged encoding
        dropout_rate_1 (float): dropout rate for first dropout layer
        dropout_rate_2 (float): dropout rate for second dropout layer
        num_hidden_layers_nt (int): number of hidden layers for nucleotide encoding
        num_labels (int): number of labels (default: 1)

    Returns:
        output (tensor): model output 
    """
    def __init__(self,
                 model_no,
                 aa_encoding_length,
                 nt_encoding_length,
                 vocab_sizes,
                 emb_size,
                 hidden_neurons_aa,
                 hidden_neurons_nt,
                 hidden_neurons_merge,
                 dropout_rate_1,
                 dropout_rate_2,
                 num_hidden_layers_nt,
                 num_labels=1):

        # Initialize the parent class
        super(NetstartModel, self).__init__()

        # Store model number
        self.model_no = model_no

        # Embedding size per taxonomical rank represented
        self.emb_size = emb_size

        # Load pretrained, finetuned model
        self.pretrained_model_aa = AutoModel.from_pretrained(
            "../../data/data_model/pretrained_models/finetuned_models/esm2-8m-finetuned_model_100u_100d_model" + str(self.model_no))

        # Freeze the pretrained model
        for param in self.pretrained_model_aa.parameters():
            param.requires_grad = False

        # Define feedforward hidden layer for protein-ness window
        self.hidden_layer_aa = nn.Linear(self.pretrained_model_aa.config.hidden_size * aa_encoding_length, hidden_neurons_aa)

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
        self.hidden_layer_2 = nn.Linear(hidden_neurons_aa + hidden_neurons_nt + emb_size, hidden_neurons_merge)

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
                x_aa,
                x_nt,
                attention_mask_aa,
                tax_ranks):
        
        """ 
        Forward pass of the model.
        
        Args:
            x_aa (tensor): amino acid encoding
            x_nt (tensor): nucleotide encoding
            attention_mask_aa (tensor): attention mask for amino acid encoding
            tax_ranks (tensor): taxonomical ranks
        """

        #Reshape the nucleotide input to flatten it (4xnucleotides to 1D tensor)
        x_nt = x_nt.view(x_nt.size(0), -1)

        #Get output from pretrained, finetuned ESM2 model
        features_aa = self.pretrained_model_aa(x_aa, attention_mask=attention_mask_aa)

        #Get entire sequence representation from last hidden state from pretrained finetuned ESM2
        sequence_output_aa = features_aa['last_hidden_state']

        #Reshape outputs to torch-size([minibatch_size, sequence_length*hidden_size])
        reshaped_output_aa = sequence_output_aa.view(sequence_output_aa.size(0), -1)

        #Define hidden layer to downscale amino acid sequence representations
        hidden_aa = self.hidden_layer_aa(reshaped_output_aa)
        hidden_aa = self.relu(hidden_aa)
        hidden_aa = self.dropout_1(hidden_aa)

        #Define hidden layer for local start codon context sequence representations
        hidden_nt = x_nt
        for layer in self.nt_layers:
            hidden_nt = layer(hidden_nt)
            hidden_nt = self.relu(hidden_nt)
            hidden_nt = self.dropout_1(hidden_nt)

        #Define embeddings for the major taxonomical ranks
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

        #Sum embeddings from each rank
        stacked_embeddings = torch.stack(embeddings_list, axis=1).sum(axis=1)
        embeddings_tax = stacked_embeddings.view(stacked_embeddings.size(0), -1)

        #Concatenate all inputs
        concatenated_input = torch.cat([hidden_aa, hidden_nt, embeddings_tax], dim=1)

        #Define hidden layer to feed all input types through together
        hidden = self.hidden_layer_2(concatenated_input)
        hidden = self.relu(hidden)
        hidden = self.dropout_2(hidden)

        #Define classification layer
        output = self.classifier(hidden)

        return output


#Define objective for hyperparameter optimization
def objective(trial, model_no):
    set_seed(42)
    torch.cuda.empty_cache()

    #Define trial suggestions
    #Define network hyperparameters
    nts_upstream = [10, 20, 30]
    nts_downstream = [10, 20, 30]
    num_hidden_neurons = [128, 256, 512, 1024, 2048]
    depth_nt_window = [1, 2, 3, 4, 5]
    dropout_rates_1 = [0.5, 0.6, 0.7]
    dropout_rates_2 = [0.3, 0.4, 0.5]
    emb_sizes = [200, 300, 400, 500, 600]
    lr_values = [4e-5, 2e-5, 1e-5, 8e-6, 6e-6, 4e-6]
    batchsize_values = [16, 32, 64] 

    emb_size_tax = trial.suggest_categorical('emb_size_tax', emb_sizes)
    hidden_neurons_aa = trial.suggest_categorical('hidden_neurons_aa', num_hidden_neurons)
    hidden_neurons_nt = trial.suggest_categorical('hidden_neurons_nt', num_hidden_neurons)
    hidden_neurons_merge = trial.suggest_categorical('hidden_neurons_merge', num_hidden_neurons)
    dropout_rate_1 = trial.suggest_categorical('dropout_rate_1', dropout_rates_1)
    dropout_rate_2 = trial.suggest_categorical('dropout_rate_2', dropout_rates_2)
    num_hidden_layers_nt = trial.suggest_categorical('depth_nt_window', depth_nt_window)
    batch_size = trial.suggest_categorical('batch_size', batchsize_values)
    lr = trial.suggest_categorical('lr', lr_values)
    extract_nts_upstream = trial.suggest_categorical('nt_upstream', nts_upstream)
    extract_nts_downstream = trial.suggest_categorical('nt_downstream', nts_downstream)
    extract_aas_upstream = 100
    extract_aas_downstream = 100

    print(f"batch_size: {batch_size}\nemb_size_tax: {emb_size_tax}\nextract_nts_upstream: {extract_nts_upstream}\nextract_nts_downstream: {extract_nts_downstream}")
    print(f"hidden_neurons_aa: {hidden_neurons_aa}\nhidden_neurons_nt: {hidden_neurons_nt}\nhidden_neurons_merge: {hidden_neurons_merge}\ndepth_nt_window: {num_hidden_layers_nt}")

    wandb.init(project=f"hyperparameter_optimization_final_{str(model_no)}", config={
        "batch_size": batch_size,
        "emb_size_tax": emb_size_tax,
        "extract_nts_upstream": extract_nts_upstream,
        "extract_nts_downstream": extract_nts_downstream,
        "hidden_neurons_aa": hidden_neurons_aa,
        "hidden_neurons_nt": hidden_neurons_nt,
        "hidden_neurons_merge": hidden_neurons_merge,
        "dropout_rate_1": dropout_rate_1,
        "dropout_rate_2": dropout_rate_2,
        "lr": lr,
        "depth_nt_window": num_hidden_layers_nt},
        name = f"trial_{trial.number}")

    #Get taxonomy mappings
    vocab_sizes, tax_mapping = load_taxonomy_mappings()

    #Get defined datasets
    df_train, df_val = extract_datasets(data_train,
                                                 data_val,
                                                 tax_mapping,
                                                 extract_upstream_nt = extract_nts_upstream,
                                                 extract_downstream_nt = extract_nts_downstream,
                                                 extract_upstream_aa = extract_aas_upstream,
                                                 extract_downstream_aa = extract_aas_downstream)


    #Get sequence and label encodings
    labels_train, train_encodings_aa, train_encodings_nt, \
    labels_val, val_encodings_aa, val_encodings_nt= create_encodings(df_train,
                                                                         df_val,
                                                                         extract_upstream_aa = extract_aas_upstream,
                                                                         extract_downstream_aa = extract_aas_downstream)

    aa_encodings_len = train_encodings_aa['input_ids'].shape[1]
    nt_encodings_len_flattened = len(train_encodings_nt[0])*len(train_encodings_nt[0][0]) #4 * numbers of nt

    #Get all datatypes for model in one dataset
    train_dataset = MultiInputDataset(train_encodings_aa, train_encodings_nt, df_train['tax_ranks_train'], labels_train)
    val_dataset = MultiInputDataset(val_encodings_aa, val_encodings_nt, df_val['tax_ranks_val'], labels_val)

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
    shuffle=True, 
    num_workers=8,  # Use fewer workers if validation data is smaller
    pin_memory=True)


    #Instantiate model
    netstart_model = NetstartModel(model_no=model_no,
                                   aa_encoding_length = aa_encodings_len,
                                   nt_encoding_length = nt_encodings_len_flattened,
                                   vocab_sizes = vocab_sizes,
                                   emb_size = emb_size_tax,
                                   hidden_neurons_aa = hidden_neurons_aa,
                                   hidden_neurons_nt = hidden_neurons_nt,
                                   hidden_neurons_merge = hidden_neurons_merge,
                                   dropout_rate_1 = dropout_rate_1,
                                   dropout_rate_2 = dropout_rate_2,
                                   num_hidden_layers_nt = num_hidden_layers_nt)

    #Determine available hardware accelerator (GPU if available, otherwise CPU) and move model to it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If multiple GPUs are available, use DataParallel
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        netstart_model = torch.nn.DataParallel(netstart_model)  # This wraps your model to use multiple GPUs

    print(f"Memory Allocated before loading model: {torch.cuda.memory_allocated(device) / 1024**3} GB")
    print(f"Running on: {device}")
    netstart_model.to(device)
    print(f"Memory Allocated after loading model: {torch.cuda.memory_allocated(device) / 1024**3} GB")

    #include class weights in loss function
    pos_weight = torch.tensor(3, dtype=torch.float).to(device)

    #Define the binary cross-entropy loss criterion
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #Define training settings
    num_epochs = 5
    optimizer = optim.Adam(netstart_model.parameters(), lr=lr)

    #Define how often to evaluate
    steps_per_epoch = len(train_dataset) / (batch_size)              #The number of steps taken per epoch
    eval_per_epoch = 8                                               #The number of evaluations made per epoch
    steps_between_evals = int(steps_per_epoch / eval_per_epoch)      #The number of steps to take between each evaluation

    print(f"Running for {num_epochs} epochs with a batchsize of {batch_size}. Total number of training sequences: {len(train_dataset)}", sep = "", flush=True)
    print(f"Evaluates every {steps_between_evals} steps.", sep = "", flush=True)

    #Initialize variables for early stopping
    best_val_loss = float('inf')
    counter_patience = 0
    step = 0

    threshold_patience = 5  #Number of evaluation cycles with no improvement (based on validation loss)

    for epoch in range(num_epochs):
        print("Epoch: ", epoch+1, flush=True)

        for batch in train_loader:
            netstart_model.train()
            
            if step % steps_between_evals == 0:
                print(f"Memory Allocated start training: {torch.cuda.memory_allocated(device) / 1024**3} GB")
            #Get inputs and labels
            inputs_aa = batch["aa_encodings"]["input_ids"].to(device)
            inputs_nt = batch["nt_encodings"].to(device)
            attention_mask_aa = batch["aa_encodings"]["attention_mask"].to(device)
            tax_ranks = batch["tax_ranks"].to(device)
            labels = batch['labels'].to(device).view(-1, 1).float()

            torch.cuda.empty_cache() 

            #Forward pass
            outputs = netstart_model(inputs_aa, inputs_nt, attention_mask_aa, tax_ranks)
            loss = loss_fn(outputs, labels)

            #Backward pass
            optimizer.zero_grad()
            loss.backward()

            #Update parameters
            optimizer.step()

            if step % steps_between_evals == 0:
                print(f"Memory Allocated end training, before clear-up: {torch.cuda.memory_allocated(device) / 1024**3} GB")

            torch.cuda.empty_cache()
            del outputs
            del loss

            if step % steps_between_evals == 0:
                print(f"Memory Allocated end training, after clear-up: {torch.cuda.memory_allocated(device) / 1024**3} GB")

            step += 1

            #Make evaluation of model
            if step % steps_between_evals == 0:
                netstart_model.eval()

                val_loss = 0.0
                val_true_labels = []
                val_predicted_probs = []

                with torch.no_grad():
                    for counter, batch in enumerate(val_loader):
                        #Get inputs and labels
                        inputs_aa = batch["aa_encodings"]["input_ids"].to(device)
                        inputs_nt = batch["nt_encodings"].to(device)
                        attention_mask_aa = batch["aa_encodings"]["attention_mask"].to(device)
                        tax_ranks = batch["tax_ranks"].to(device)
                        labels = batch['labels'].to(device).view(-1, 1).float()

                        #Forward pass
                        outputs = netstart_model(inputs_aa, inputs_nt, attention_mask_aa, tax_ranks)
                        loss = loss_fn(outputs, labels)

                        val_loss += loss.item() #get mean loss of batch

                        val_true_labels.extend(labels.cpu().numpy().flatten())
                        val_predicted_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                        
                        #Validate on the first 1M validation samples (shuffled)
                        if counter*batch_size >= 1000000:
                            no_batches_val = counter + 1
                            break
                        else:
                            no_batches_val = counter + 1

                    val_avg_loss = round(val_loss / no_batches_val, 8)

                    print(f"Memory Allocated end validation, before clear-up: {torch.cuda.memory_allocated(device) / 1024**3} GB")

                    del loss
                    del outputs
                    torch.cuda.empty_cache()

                    print(f"Memory Allocated end validation, after clear-up: {torch.cuda.memory_allocated(device) / 1024**3} GB")

                    print(f"ROC score: {roc_auc_score(val_true_labels, val_predicted_probs)}", flush = True)

                    wandb.log({
                        "val_loss": val_avg_loss,
                        "roc_auc_score": roc_auc_score(val_true_labels, val_predicted_probs)
                    })

                    #Implement pruning strategy
                    trial.report(val_avg_loss, step)

                    #Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    #Check validation loss, apply early stopping if necessary
                    if val_avg_loss < best_val_loss:

                        #Update lowest loss
                        best_val_loss = val_avg_loss

                        #Reset counter
                        counter_patience = 0

                    else:
                        counter_patience += 1

                #When the model is overfitting, use early stopping
                if counter_patience >= threshold_patience:
                    break


        #When the model is overfitting, use early stopping
        if counter_patience >= threshold_patience:
            break

    wandb.finish()
    return val_avg_loss #Return objective to evaluate hyperparameters on


####Main code####
#Define datasplit
data_train, data_val = load_and_partition_data(model_no = model_no)
#Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize',   #Minimize loss
                            pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=30)  #Runs 30 trials to find the best hyperparameters

#Get the best hyperparameters from the study
best_up_nt = study.best_params['nt_upstream']
best_down_nt = study.best_params['nt_downstream']
best_lr = study.best_params['lr']
best_bs = study.best_params['batch_size']
best_neu_aa = study.best_params['hidden_neurons_aa']
best_neu_nt = study.best_params['hidden_neurons_nt']
best_neu_merge = study.best_params['hidden_neurons_merge']
best_drop1 = study.best_params['dropout_rate_1']
best_drop2 = study.best_params['dropout_rate_2']
best_emb = study.best_params['emb_size_tax']
best_depth_nt_window = study.best_params['depth_nt_window']

##Save hyperparameters##
model_config = {
    "model_no": model_no,
    "nt_upstream": best_up_nt,
    "nt_downstream": best_down_nt,
    "batch_size": best_bs,
    "emb_size_tax": best_emb,
    "hidden_neurons_aa": best_neu_aa,
    "hidden_neurons_nt": best_neu_nt,
    "hidden_neurons_merge": best_neu_merge,
    "dropout_rate_1": best_drop1,
    "dropout_rate_2": best_drop2,
    "lr": best_lr,
    "depth_nt_window": best_depth_nt_window}

#Save the dictionary as a JSON file
file_path = f'../../data/data_model/hyperparameters/netstart_model{model_no}_hyperparameters.json'
with open(file_path, 'w') as json_file:
    json.dump(model_config, json_file, indent=4)

print(f"Model configuration saved to {file_path}.")