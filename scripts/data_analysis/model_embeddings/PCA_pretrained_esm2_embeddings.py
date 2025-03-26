import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from transformers import AutoTokenizer, AutoModel

def encode_nucleotide_to_amino_acid(sequence):
    """
    The function takes a nucleotide sequence and translates it into the corresponding amino acid sequence

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
        amino_acid = genetic_code.get(codon, "<pad>")  # * represents unknown
        amino_acid_sequence += amino_acid

    return amino_acid_sequence


def extract_datasets(data_test,
                     extract_upstream_aa,
                     extract_downstream_aa):
    """
    Create the datasets to use for model development

    Args:
        data_test (df): the loaded dataframe with test partition
        extract_upstream_nt (int): the number of nucleotide positions upstream ATG to extract
        extract_downstream_nt (int): the number of nucleotide positions downstream ATG to extract
        extract_upstream_aa (int): the number of amino acid positions upstream ATG to extract
        extract_downstream_aa (int): the number of amino acid positions downstream ATG to extract
    """

    #Initialize
    aa_sequences_test = []      #x test
    labels_test = []            #y test
    seq_type_test = []
    species = []

    ATG_position = 300
    counter = 0

    data_test = data_test.sample(n=20000, random_state=42)


    #Extract sequences from test dataset
    for row in data_test.itertuples():
        counter += 1

        assert row.Sequence[ATG_position:ATG_position+3] == "ATG"

        aa_sequences_test.append(encode_nucleotide_to_amino_acid(row.Sequence[ATG_position-3*extract_upstream_aa:ATG_position+3+3*extract_downstream_aa]))
        labels_test.append(float(row.TIS))
        seq_type_test.append(row.sequence_type)
        species.append(row.Species)

    return aa_sequences_test, labels_test, seq_type_test, species

class MultiInputDataset(torch.utils.data.Dataset):
    """
    This class is used to create a dataset for inputting to the model with multiple types of inputs
    """
    def __init__(self, encodings, labels, seq_type_test, species):
        self.encodings = encodings
        self.labels = labels
        self.seq_type_test = seq_type_test
        self.species = species

    def __getitem__(self, idx):
        item = {
            'encodings': {key: torch.as_tensor(val[idx]) for key, val in self.encodings.items()},
            'labels': torch.as_tensor(self.labels[idx]),
            'seq_type_test': self.seq_type_test[idx],
            'species': self.species[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)


class PretrainedProteinLM(nn.Module):
    """
    Get sequence representation from ESM-2.
    """
    def __init__(self, model_name):
        super(PretrainedProteinLM, self).__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)

    def forward(self, x):
        #Pass input through the pretrained model to extract features
        features = self.pretrained_model(x)

        #Use the entire sequence representation
        sequence_output = features['last_hidden_state']

        return sequence_output 


def run_pca(model_name, tokenizer_name, finetuned):
    """
    Executes pipeline of extracting data and performing PCA.

    Args:
        model_name (str): path to file of language model to perform PCA on.
        tokenizer_name (str): the name of the corresponding tokenizer.
        finetuned (True/False): specify whether the language model to perform PCA on is finetuned or not.
    """
    print("Started running")
    model_name_clean = str(model_name.split("/")[-1])
    aa_sequences_test, labels_test, seq_type_test, species = extract_datasets(
                         data_test,
                         extract_upstream_aa = 100,
                         extract_downstream_aa = 100)

    print("Running analyses for: ", model_name_clean, flush = True)

    print("Testing samples: ", len(labels_test), flush = True)

    #Encode labels to ensure correct format
    encoder = LabelEncoder()

    encoder.fit(labels_test)
    labels_test = encoder.transform(labels_test)

    aa_seqs_len = 100 + 1 + 100

    #load model and encodings
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
        do_lower_case=False, 
        model_max_length=aa_seqs_len + 1)

        
    test_encodings = tokenizer(aa_sequences_test, 
                                   padding=True,  
                                   truncation=True,
                                   return_tensors="pt")
    
    attention_mask_test = test_encodings['input_ids'] != tokenizer.pad_token_id
    test_encodings['attention_mask'] = attention_mask_test.int()  #Replace the old attention mask

    model = PretrainedProteinLM(model_name = model_name)

    test_dataset = MultiInputDataset(test_encodings, labels_test, seq_type_test, species)
    
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(device, flush = True)
    model.to(device)

    batch_size = 64

    #Define data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    embeddings_numpy = []  
    labels_list = []     
    seq_types_list = []
    species_list = []

    for batch in test_loader:
        #Get inputs and labels
        inputs = batch["encodings"]["input_ids"].to(device)

        labels = batch['labels'].to(device).view(-1, 1).float()

        seq_types = batch['seq_type_test']
        species = batch['species']

        #Calculate model outputs
        outputs = model(inputs)

        #Append current batch embeddings and labels to the respective lists
        embeddings_tensor = outputs
        embeddings_reshaped = embeddings_tensor.view(embeddings_tensor.size(0), -1)
        embeddings_batch_numpy = embeddings_reshaped.detach().cpu().numpy()

        embeddings_numpy.append(embeddings_batch_numpy)
        labels_list.extend(labels.squeeze().tolist())
        seq_types_list.extend(seq_types)
        species_list.extend(species)

    #Concatenate embeddings from all minibatches along the first axis (batch dimension)
    embeddings_numpy = np.concatenate(embeddings_numpy, axis=0)

    #perform standardization
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings_numpy)

    #perform PCA
    pca = PCA(n_components=2)  # Choose the number of components (e.g., 2 or 3)
    embeddings_pca = pca.fit_transform(normalized_embeddings)

    #Calculate the variance explained by the first two principal components
    variance_explained = pca.explained_variance_ratio_
    print("Variance explained by PC1:", variance_explained[0], flush = True)
    print("Variance explained by PC2:", variance_explained[1], flush = True)

    #Get the total number of principal components
    total_components = pca.n_components_
    print("Total number of principal components:", total_components, flush = True)

    #Filter the matrix based on TIS/non-TIS labels
    pca_label_0 = [row for idx, row in enumerate(embeddings_pca) if labels_list[idx] == 0.0]
    pca_label_1 = [row for idx, row in enumerate(embeddings_pca) if labels_list[idx] == 1.0]

    #Splitting the arrays into two separate lists
    pca_label_0_PC1 = [row[0] for row in pca_label_0]  
    pca_label_0_PC2 = [row[1] for row in pca_label_0]  

    pca_label_1_PC1 = [row[0] for row in pca_label_1] 
    pca_label_1_PC2 = [row[1] for row in pca_label_1]  

    title = "Pretrained ESM-2"
    if finetuned == True:
            title += ", "
            title += "Finetuned"

    colors = ['tab:blue','tab:green']

    plt.figure(figsize= [8,8])
    plt.scatter(pca_label_0_PC1, pca_label_0_PC2, color=colors[0], alpha=0.8, s=18)
    plt.scatter(pca_label_1_PC1, pca_label_1_PC2, color=colors[1], alpha=0.8, s=18)
    plt.xlabel('PC1 ('+str(round(variance_explained[0]*100, 2))+'% variance explained)', fontsize = 15)
    plt.ylabel('PC2 ('+str(round(variance_explained[1]*100, 2))+'% variance explained)', fontsize = 15)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.title(title, fontsize = 20)
    if finetuned == False:
        plt.legend(["non-TIS", "TIS"])
    plt.savefig("../../../results/learned_model_embeddings/esm2_embeddings/PCA_binary_"+model_name_clean+".png")

    #Plot Sequence categories
    #'Intergenic', 'TIS', 'Intron'
    #'Downstream, out of frame', 'Downstream, in frame', 'Upstream, in frame', 'Upstream, out of frame'
        
    pca_upstream_out = [row for idx, row in enumerate(embeddings_pca) if seq_types_list[idx] == 'Upstream, out of frame']
    pca_upstream_in = [row for idx, row in enumerate(embeddings_pca) if seq_types_list[idx] == 'Upstream, in frame']
    pca_downstream_out = [row for idx, row in enumerate(embeddings_pca) if seq_types_list[idx] == 'Downstream, out of frame']
    pca_downstream_in = [row for idx, row in enumerate(embeddings_pca) if seq_types_list[idx] == 'Downstream, in frame']
    pca_intergenic = [row for idx, row in enumerate(embeddings_pca) if seq_types_list[idx] == 'Intergenic']
    pca_introns = [row for idx, row in enumerate(embeddings_pca) if seq_types_list[idx] == 'Intron']
    pca_TIS = [row for idx, row in enumerate(embeddings_pca) if seq_types_list[idx] == 'TIS']

    pca_upstream_out_PC1 = [row[0] for row in pca_upstream_out]  
    pca_upstream_out_PC2 = [row[1] for row in pca_upstream_out]  

    pca_upstream_in_PC1 = [row[0] for row in pca_upstream_in]  
    pca_upstream_in_PC2 = [row[1] for row in pca_upstream_in]  
        
    pca_downstream_out_PC1 = [row[0] for row in pca_downstream_out]  
    pca_downstream_out_PC2 = [row[1] for row in pca_downstream_out]  

    pca_downstream_in_PC1 = [row[0] for row in pca_downstream_in]  
    pca_downstream_in_PC2 = [row[1] for row in pca_downstream_in]  
        
    pca_intergenic_PC1 = [row[0] for row in pca_intergenic]  
    pca_intergenic_PC2 = [row[1] for row in pca_intergenic]  
        
    pca_introns_PC1 = [row[0] for row in pca_introns]  
    pca_introns_PC2 = [row[1] for row in pca_introns]  
        
    pca_TIS_PC1 = [row[0] for row in pca_TIS]  
    pca_TIS_PC2 = [row[1] for row in pca_TIS]  

    title = "Pretrained ESM-2"
    if finetuned == True:
            title += ", "
            title += "Finetuned"
    
    colors = sns.color_palette("tab10")
        
    plt.figure(figsize= [8,8])
    plt.scatter(pca_intergenic_PC1, pca_intergenic_PC2, color=colors[0], alpha=1, s=18)
    plt.scatter(pca_introns_PC1, pca_introns_PC2, color=colors[5], alpha=0.9, s=18)
    plt.scatter(pca_upstream_out_PC1, pca_upstream_out_PC2, color=colors[3], alpha=0.8, s=18)
    plt.scatter(pca_upstream_in_PC1, pca_upstream_in_PC2, color=colors[1], alpha=0.7, s=18)
    plt.scatter(pca_downstream_out_PC1, pca_downstream_out_PC2, color=colors[9], alpha=0.7, s=18)
    plt.scatter(pca_downstream_in_PC1, pca_downstream_in_PC2, color=colors[6], alpha=0.7, s=18)
    plt.scatter(pca_TIS_PC1, pca_TIS_PC2, color=colors[2], alpha=0.8, s=18)
        
    plt.xlabel('PC1 ('+str(round(variance_explained[0]*100, 2))+'% variance explained)', fontsize = 16)
    plt.ylabel('PC2 ('+str(round(variance_explained[1]*100, 2))+'% variance explained)', fontsize = 16)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.title(title, fontsize = 18)
    #if finetuned == False:
    #    plt.legend(['Intergenic', 'Intron', 
    #                'Upstream, out of frame', 'Upstream, in frame', 
    #                'Downstream, out of frame', 'Downstream, in frame',
    #                'TIS'])
    plt.savefig("../../../results/learned_model_embeddings/esm2_embeddings/PCA_seq_types_"+model_name_clean+".png")


###Main code###
#Read the compressed CSV files for each partition into df
data_test = pd.read_csv("../../../data/data_model/datasets/data_partition_5.csv.gz", compression='gzip')

#Pretrained model, not finetuned (8M parameters)
run_pca(model_name = "facebook/esm2_t6_8M_UR50D", 
        tokenizer_name = "facebook/esm2_t6_8M_UR50D", 
        finetuned = False)


#Pretrained models, finetuned (8M parameters)
run_pca(model_name = "../../../data/data_model/pretrained_models/finetuned_models/esm2-8m-finetuned_model_100u_100d_model1", 
        tokenizer_name = "facebook/esm2_t6_8M_UR50D", 
        finetuned = True)

run_pca(model_name = "../../../data/data_model/pretrained_models/finetuned_models/esm2-8m-finetuned_model_100u_100d_model2", 
        tokenizer_name = "facebook/esm2_t6_8M_UR50D", 
        finetuned = True)
      
run_pca(model_name = "../../../data/data_model/pretrained_models/finetuned_models/esm2-8m-finetuned_model_100u_100d_model3", 
        tokenizer_name = "facebook/esm2_t6_8M_UR50D", 
        finetuned = True)

run_pca(model_name = "../../../data/data_model/pretrained_models/finetuned_models/esm2-8m-finetuned_model_100u_100d_model4", 
        tokenizer_name = "facebook/esm2_t6_8M_UR50D", 
        finetuned = True)
