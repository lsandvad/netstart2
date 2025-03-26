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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

#train all 4 models with the optimized, respective hyperparameters
model_no = 1
hyperpars_filename = 'netstart_model'+str(model_no)+'_hyperparameters.json'
esm2_pretrained = "esm2-8m-finetuned_model_100u_100d_model" + str(model_no) 
netstart_model_filename = "netstart_model" + str(model_no)

#Collect optimized hyperparameters
with open("../../data/data_model/hyperparameters/"+hyperpars_filename, 'r') as json_file:
    model_config = json.load(json_file)

#Define hyperparameters
nt_upstream = model_config["nt_upstream"]
nt_downstream = model_config["nt_downstream"]
batch_size = model_config["batch_size"]
emb_size_tax = model_config["emb_size_tax"]
hidden_neurons_aa = model_config["hidden_neurons_aa"]
hidden_neurons_nt = model_config["hidden_neurons_nt"]
hidden_neurons_merge = model_config["hidden_neurons_merge"]
dropout_rate_1 = model_config["dropout_rate_1"]
dropout_rate_2 = model_config["dropout_rate_2"]
depth_nt_window = model_config["depth_nt_window"]

# Adjust the learning rate by multiplying it by 0.75 (uses already trained model, fine-tune on group specific data)
lr = model_config["lr"] * 0.75

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

def load_and_partition_data(model_no, org_group):
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

    elif model_no == 4:
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

    data_train_org = data_train[data_train["group"] == org_group]
    data_val_org = data_val[data_val["group"] == org_group]
    data_test_org = data_test[data_test["group"] == org_group]

    return data_train_org, data_val_org, data_test_org


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


def load_taxonomy_mappings():
    """
    Load major taxonomy ranks (species, genus, family, order, class, phylum, kingdom)
    for each species in dataset.

    Returns:
        vocab_sizes (list): A list of integers representing the number of different taxonomic groups 
                            present in each rank [species, genus, family, order, class, phylum, kingdom].
        tax_mapping (dict): A dictionary where keys are species names and values are lists of taxonomic 
                            IDs for each rank [species_id, genus_id, family_id, order_id, class_id, phylum_id, kingdom_id].
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
                     data_test, 
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
        data_test (df): the loaded dataframe with test partition
        tax_mapping (dict): dictionary with mappings of taxonomic ranks for each species
        extract_upstream_nt (int): the number of nucleotide positions upstream ATG to extract
        extract_downstream_nt (int): the number of nucleotide positions downstream ATG to extract
        extract_upstream_aa (int): the number of amino acid positions upstream ATG to extract
        extract_downstream_aa (int): the number of amino acid positions downstream ATG to extract
    """
    
    rows = []

    for i, row in enumerate(data_train.itertuples()):
        taxonomy_levels = tax_mapping[row.Species]
        ATG_position = row.Sequence.find("ATG")
        assert row.Sequence[ATG_position:ATG_position+3] == "ATG", f"Expected 'ATG' at position {ATG_position}, but found {row.Sequence[ATG_position:ATG_position+3]}"

        #Append each row of wanted information to the list
        rows.append({
            'aa_sequences_train': encode_nucleotide_to_amino_acid(row.Sequence[ATG_position-3*extract_upstream_aa:ATG_position+3+3*extract_downstream_aa]),
            'nt_sequences_train': row.Sequence[ATG_position-extract_upstream_nt:ATG_position+3+extract_downstream_nt],
            'tax_ranks_train': np.array(taxonomy_levels),
            'group_train': row.group,
            'labels_train': int(row.TIS)
        })
    
    #Store data in df
    df_train = pd.DataFrame(rows)
    
    rows = []
    for i, row in enumerate(data_val.itertuples()):
        taxonomy_levels = tax_mapping[row.Species] 
        assert row.Sequence[ATG_position:ATG_position+3] == "ATG", print(row.Sequence[ATG_position:ATG_position+3])
        
        rows.append({
            'aa_sequences_val': encode_nucleotide_to_amino_acid(row.Sequence[ATG_position-3*extract_upstream_aa:ATG_position+3+3*extract_downstream_aa]),
            'nt_sequences_val': row.Sequence[ATG_position-extract_upstream_nt:ATG_position+3+extract_downstream_nt],
            'tax_ranks_val': np.array(taxonomy_levels),
            'group_val': row.group,
            'labels_val': int(row.TIS)
        })
            
    df_val = pd.DataFrame(rows)
    
    rows = []
    for i, row in enumerate(data_test.itertuples()):
        taxonomy_levels = tax_mapping[row.Species]
        
        assert row.Sequence[ATG_position:ATG_position+3] == "ATG", print(row.Sequence[ATG_position:ATG_position+3])

        rows.append({
            'aa_sequences_test': encode_nucleotide_to_amino_acid(row.Sequence[ATG_position-3*extract_upstream_aa:ATG_position+3+3*extract_downstream_aa]),
            'nt_sequences_test': row.Sequence[ATG_position-extract_upstream_nt:ATG_position+3+extract_downstream_nt],
            'tax_ranks_test': np.array(taxonomy_levels),
            'group_test': row.group,
            'labels_test': int(row.TIS)
        })

    df_test = pd.DataFrame(rows)

    return df_train, df_val, df_test


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
        # For 'N', we do nothing, so the corresponding column remains all zeros
    
    return encoding

def create_encodings(df_train, df_val, df_test, extract_upstream_aa, extract_downstream_aa, tokenizer_aa):
    """
    Encode labels and sequences to fit input to pretrained models.

    Args:
        df_train (dataframe): dataframe with training data
        df_val (dataframe): dataframe with validation data
        df_test (dataframe): dataframe with test data
        tokenizer_aa (AutoTokenizer): preloaded tokenizer for amino acid sequences
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

    #Encode nucleotide sequences
    #One-hot encode and format all sequences in dataset
    sequences_train = [one_hot_encode(seq) for seq in df_train['nt_sequences_train']]
    sequences_train_nt = torch.stack(sequences_train)
    sequences_val = [one_hot_encode(seq) for seq in df_val['nt_sequences_val']]
    sequences_val_nt = torch.stack(sequences_val)
    sequences_test = [one_hot_encode(seq) for seq in df_test['nt_sequences_test']]
    sequences_test_nt = torch.stack(sequences_test)
    
    return labels_train, train_encodings_aa, sequences_train_nt, \
           labels_val, val_encodings_aa, sequences_val_nt, \
           labels_test, test_encodings_aa, sequences_test_nt


class MultiInputDataset(torch.utils.data.Dataset):
    """
    This class is used to create a dataset for inputting to the model with multiple types of inputs.
    """
    def __init__(self, aa_encodings, nt_encodings, tax_ranks, group, labels):
        self.aa_encodings = aa_encodings
        self.nt_encodings = nt_encodings
        self.tax_ranks = tax_ranks
        self.group = group
        self.labels = labels

    def __getitem__(self, idx):
        #Convert strings to integers in the 'tax_ranks' list
        tax_ranks_item = [int(rank) if rank != '0' else 0 for rank in self.tax_ranks[idx]]
        
        item = {
            'aa_encodings': {key: torch.as_tensor(val[idx]) for key, val in self.aa_encodings.items()},
            'nt_encodings': torch.as_tensor(self.nt_encodings[idx], dtype=torch.float32),
            'tax_ranks': torch.LongTensor(tax_ranks_item),
            'group': self.group[idx],
            'labels': torch.as_tensor(self.labels[idx])
        }
        return item

    def __len__(self):
        return len(self.labels)


class NetstartModel(nn.Module):
    """
    Define model architecture.
    """
    def __init__(self, 
                 aa_encoding_length, 
                 nt_encoding_length, 
                 vocab_sizes, 
                 emb_size = emb_size_tax,
                 hidden_neurons_aa = hidden_neurons_aa,
                 hidden_neurons_nt = hidden_neurons_nt,
                 hidden_neurons_merge = hidden_neurons_merge,
                 dropout_rate_1 = dropout_rate_1,
                 dropout_rate_2 = dropout_rate_2,
                 num_hidden_layers_nt = depth_nt_window,
                 num_labels = 1):
        
        super(NetstartModel, self).__init__()
        
        #Embedding size per taxonomical rank represented
        self.emb_size = emb_size

        ###Define layers###
        self.pretrained_model_aa = AutoModel.from_pretrained("../../data/data_model/pretrained_models/finetuned_models/"+esm2_pretrained)
    
        #Dont update weights on finetuned models
        for param in self.pretrained_model_aa.parameters():
            param.requires_grad = True
        
        self.hidden_layer_aa = nn.Linear(self.pretrained_model_aa.config.hidden_size*aa_encoding_length, hidden_neurons_aa)

        #Define feedforward hidden layers for local start codon context window
        self.nt_layers = nn.ModuleList()
        self.nt_layers.append(nn.Linear(nt_encoding_length, hidden_neurons_nt))  # First layer

        #Additional hidden layers for nt encoding if `num_hidden_layers_nt > 1`
        for _ in range(num_hidden_layers_nt - 1):
            self.nt_layers.append(nn.Linear(hidden_neurons_nt, hidden_neurons_nt))

        #Define taxonomical embeddings
        self.embedding_species = nn.Embedding(vocab_sizes[0], emb_size, padding_idx=0)
        self.embedding_genus = nn.Embedding(vocab_sizes[1], emb_size, padding_idx=0)
        self.embedding_family = nn.Embedding(vocab_sizes[2], emb_size, padding_idx=0)
        self.embedding_order = nn.Embedding(vocab_sizes[3], emb_size, padding_idx=0)
        self.embedding_class = nn.Embedding(vocab_sizes[4], emb_size, padding_idx=0)
        self.embedding_phylum = nn.Embedding(vocab_sizes[5], emb_size, padding_idx=0)
        self.embedding_kingdom = nn.Embedding(vocab_sizes[6], emb_size, padding_idx=0)
        
        self.hidden_layer_2 = nn.Linear(hidden_neurons_aa+hidden_neurons_nt+emb_size, hidden_neurons_merge)

        self.classifier = nn.Linear(hidden_neurons_merge, num_labels)
        
        #Define dropout
        self.dropout_1 = nn.Dropout(dropout_rate_1)
        self.dropout_2 = nn.Dropout(dropout_rate_2)

        #Define binary output
        self.num_labels = num_labels
        
        #Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, 
                x_aa, 
                x_nt, 
                attention_mask_aa, 
                tax_ranks):
        
        x_nt = x_nt.view(x_nt.size(0), -1)  #Reshape the input to flatten it (4x23 to 1D tensor)
        features_aa = self.pretrained_model_aa(x_aa, attention_mask = attention_mask_aa)
        
        #Get entire sequence representation from last hidden state from both pretrained LMs
        sequence_output_aa = features_aa['last_hidden_state']

        #Reshape outputs to torch-size([minibatch_size, sequence_length*hidden_size])
        reshaped_output_aa = sequence_output_aa.view(sequence_output_aa.size(0), -1)
        
        #define hidden layer to downscale amino acid sequence representations
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
        
        #sum embeddings from each rank
        stacked_embeddings = torch.stack(embeddings_list, axis=1).sum(axis=1)
        embeddings_tax = stacked_embeddings.view(stacked_embeddings.size(0), -1)

        #concatenate all inputs
        concatenated_input = torch.cat([hidden_aa, hidden_nt, embeddings_tax], dim=1)
        
        #Define hidden layer to feed all input types through together
        #Pass the pooled output through the downstream FFN
        hidden = self.hidden_layer_2(concatenated_input)
        hidden = self.relu(hidden)
        hidden = self.dropout_2(hidden) 

        output = self.classifier(hidden)
        #output = self.sigmoid(output)
        
        return output


###Main code###
def main(org_group, model_no):
    #Load Data
    data_train, data_val, data_test = load_and_partition_data(model_no = model_no, org_group = org_group)

    #Load Taxonomy Mappings
    vocab_sizes, tax_mapping = load_taxonomy_mappings()

    #Get data
    df_train, df_val, df_test = extract_datasets(data_train, 
                                                data_val, 
                                                data_test, 
                                                tax_mapping,
                                                extract_upstream_nt = nt_upstream, 
                                                extract_downstream_nt = nt_downstream, 
                                                extract_upstream_aa = aa_upstream,  
                                                extract_downstream_aa = aa_downstream)

    # Instantiate the tokenizer outside the function
    tokenizer_aa = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", 
                                          do_lower_case=False, 
                                          model_max_length=aa_upstream + 1 + aa_downstream + 2)

    #Get sequence and label encodings 
    labels_train, train_encodings_aa, train_encodings_nt, \
    labels_val, val_encodings_aa, val_encodings_nt, \
    labels_test, test_encodings_aa, test_encodings_nt = create_encodings(df_train, 
                                                                        df_val, 
                                                                        df_test, 
                                                                        extract_upstream_aa = aa_upstream, 
                                                                        extract_downstream_aa = aa_downstream,
                                                                        tokenizer_aa = tokenizer_aa)


    aa_encodings_len = train_encodings_aa['input_ids'].shape[1]
    nt_encodings_len_flattened = len(train_encodings_nt[0])*len(train_encodings_nt[0][0]) #4 * numbers of nt


    #Get all datatypes for model in one dataset
    train_dataset = MultiInputDataset(train_encodings_aa, train_encodings_nt, df_train['tax_ranks_train'], df_train["group_train"], labels_train)
    val_dataset = MultiInputDataset(val_encodings_aa, val_encodings_nt, df_val['tax_ranks_val'], df_val["group_val"], labels_val)
    test_dataset = MultiInputDataset(test_encodings_aa, test_encodings_nt, df_test['tax_ranks_test'], df_test["group_test"], labels_test)

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


    checkpoint = torch.load('../../data/data_model/models/netstart_model'+str(model_no)+'.pth', map_location=torch.device('cpu'), weights_only = True)

    # Remove 'module.' prefix if present and assigning the entire loaded dictionary as the state dictionary
    model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    #Instantiate model
    netstart_model = NetstartModel(aa_encoding_length = aa_encodings_len, 
                                nt_encoding_length = nt_encodings_len_flattened, 
                                vocab_sizes = vocab_sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If multiple GPUs are available, use DataParallel
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        netstart_model = torch.nn.DataParallel(netstart_model)  # This wraps your model to use multiple GPUs

    netstart_model.to(device)

    #Instantiate model with checkpoint
    netstart_model.load_state_dict(model_state_dict)

    # The pos_weight tensor is used to handle class imbalance by assigning a higher weight to the positive class in the loss function.
    pos_weight = torch.tensor(3, dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #Define training settings
    num_epochs = 5
    optimizer = optim.Adam(netstart_model.parameters(), lr=lr)

    #Define how often to evaluate
    steps_per_epoch = len(train_dataset) / (batch_size)                             #The number of steps taken per epoch
    eval_per_epoch = 12                                                              #The number of evaluations made per epoch
    steps_between_evals = int(steps_per_epoch / eval_per_epoch)                     #The number of steps to take between each evaluation

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
        project="train_netstart_final_groups",

        config = {
            "model_number": str(model_no)
        }
    )


    print("Initializing training", flush = True)
    #Training loop
    for epoch in range(num_epochs):
        print("Epoch: ", epoch+1, flush=True)

        for batch in train_loader:
            netstart_model.train()
            #Get inputs and labels
            inputs_aa = batch["aa_encodings"]["input_ids"].to(device)
            inputs_nt = batch["nt_encodings"].to(device)
            attention_mask_aa = batch["aa_encodings"]["attention_mask"].to(device)
            tax_ranks = batch["tax_ranks"].to(device)
            labels = batch['labels'].to(device).view(-1, 1).float()

            #Forward pass
            outputs = netstart_model(inputs_aa, inputs_nt, attention_mask_aa, tax_ranks)
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

            if step % int(steps_between_evals/2) == 0:
                print(f"Step {step} / {int(steps_per_epoch)*5}", sep = " ", flush=True)

            # Make evaluation of model
            if step % steps_between_evals == 0:
                netstart_model.eval()

                val_loss = 0.0
                val_true_labels = []
                val_predicted_probs = []

                # Group-wise dictionaries
                groupwise_true_labels = defaultdict(list)
                groupwise_predicted_probs = defaultdict(list)
                groupwise_losses = defaultdict(float)

                with torch.no_grad():
                    for batch in val_loader:
                        # Get inputs and labels
                        inputs_aa = batch["aa_encodings"]["input_ids"].to(device)
                        inputs_nt = batch["nt_encodings"].to(device)
                        attention_mask_aa = batch["aa_encodings"]["attention_mask"].to(device)
                        tax_ranks = batch["tax_ranks"].to(device)
                        group = batch["group"]  # Organism group
                        labels = batch['labels'].to(device).view(-1, 1).float()

                        # Forward pass
                        outputs = netstart_model(inputs_aa, inputs_nt, attention_mask_aa, tax_ranks)
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
                    val_auc_roc = round(roc_auc_score(val_true_labels, val_predicted_probs), 4)
                    precision, recall, _ = precision_recall_curve(val_true_labels, val_predicted_probs)
                    val_avg_precision = average_precision_score(val_true_labels, val_predicted_probs)
                    val_pr_auc = round(auc(recall, precision), 4)

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
                            f"{g}_val_loss": group_loss,
                            f"{g}_Val AUROC": group_auroc,
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
                        if isinstance(netstart_model, torch.nn.DataParallel) or isinstance(netstart_model, torch.nn.parallel.DistributedDataParallel):
                            state_dict = netstart_model.module.state_dict()  # Strips 'module.' prefix
                        else:
                            state_dict = netstart_model.state_dict()

                        # Save the model parameters
                        torch.save(state_dict, '../../data/data_model/models/' + netstart_model_filename + "_" + org_group + ".pth")
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
            inputs_nt = batch["nt_encodings"].to(device)
            attention_mask_aa = batch["aa_encodings"]["attention_mask"].to(device)
            tax_ranks = batch["tax_ranks"].to(device)
            labels = batch['labels'].to(device).view(-1, 1).float()

            #Get predctions and loss
            outputs = netstart_model(inputs_aa, inputs_nt, attention_mask_aa, tax_ranks)

            # Append true labels and predicted probabilities
            test_true_labels.extend(labels.cpu().numpy().flatten())
            test_predicted_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())

            
    ##Plot performance at different MCC thresholds
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

    test_true_labels = np.array(test_true_labels)
    test_predicted_probs = np.array(test_predicted_probs)
    predicted_labels = (test_predicted_probs >= best_threshold).astype(int)

    TP = np.sum((test_true_labels == 1) & (predicted_labels == 1))
    FN = np.sum((test_true_labels == 1) & (predicted_labels == 0))
    TN = np.sum((test_true_labels == 0) & (predicted_labels == 0))
    FP = np.sum((test_true_labels == 0) & (predicted_labels == 1))

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    #Calculate Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(np.array(test_true_labels), (np.array(test_predicted_probs) >= best_threshold).astype(int))

    assert total == TP + FN + TN + FP, "Something went wrong calculating test performance"

    print("Test Metrics:", flush = True)
    print(f"Test Accuracy: {test_accuracy:.4f}", flush = True)
    print(f"Test ROC AUC: {test_roc_auc:.4f}, \nTest PR AUC: {test_pr_auc:.4f}", flush = True)
    print(f"Test Sensitivity: {sensitivity:.4f}, \nTest Specificity: {specificity:.4f}\nTest MCC: {mcc:.4f}", flush = True)


groups = ["protozoa", "fungi", "invertebrate", "plant", "vertebrate"]

for org_group in groups:
    print(f"Training on data from {org_group}.")
    main(org_group, model_no)