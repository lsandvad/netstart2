#!/usr/bin/env python3.8

import time
import numpy as np
import pandas as pd
import json
import gzip
from tqdm import tqdm
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import transformers
import time
import argparse
import os 
import sys

from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

transformers.logging.set_verbosity_error()

#Define dictionary for species and phyla
conversion_dict = {
    'a_mississippiensis': 'Alligator mississippiensis',
    'a_carolinensis': 'Anolis carolinensis',
    'a_gambiae': 'Anopheles gambiae',
    'a_mellifera': 'Apis mellifera',
    'a_thaliana': 'Arabidopsis thaliana',
    'a_nidulans': 'Aspergillus nidulans',
    'b_taurus': 'Bos taurus',
    'b_distachyon': 'Brachypodium distachyon',
    'c_elegans': 'Caenorhabditis elegans',
    'c_lupus': 'Canis lupus',
    'c_livia': 'Columba livia',
    'c_cinerea': 'Coprinopsis cinerea',
    'c_neoformans': 'Cryptococcus neoformans',
    'd_rerio': 'Danio rerio',
    'd_carinata': 'Daphnia carinata',
    'd_discoideum': 'Dictyostelium discoideum',
    'd_melanogaster': 'Drosophila melanogaster',
    'e_maxima': 'Eimeria maxima',
    'e_histolytica': 'Entamoeba histolytica',
    'e_caballus': 'Equus caballus',
    'g_gallus': 'Gallus gallus',
    'g_intestinalis': 'Giardia intestinalis',
    'g_max': 'Glycine max',
    'g_gorilla': 'Gorilla gorilla',
    'h_sapiens': 'Homo sapiens',
    'h_vulgare': 'Hordeum vulgare',
    'l_donovani': 'Leishmania donovani',
    'l_japonicus': 'Lotus japonicus',
    'm_sexta': 'Manduca sexta',
    'm_truncatula': 'Medicago truncatula',
    'm_musculus': 'Mus musculus',
    'n_crassa': 'Neurospora crassa',
    'n_tabacum': 'Nicotiana tabacum',
    'o_niloticus': 'Oreochromis niloticus',
    'o_cuniculus': 'Oryctolagus cuniculus',
    'o_sativa': 'Oryza sativa',
    'o_latipes': 'Oryzias latipes',
    'o_aries': 'Ovis aries',
    'p_troglodytes': 'Pan troglodytes',
    'p_dactylifera': 'Phoenix dactylifera',
    'p_falciparum': 'Plasmodium falciparum',
    'r_norvegicus': 'Rattus norvegicus',
    'r_irregularis': 'Rhizophagus irregularis',
    's_cerevisiae': 'Saccharomyces cerevisiae',
    's_commune': 'Schizophyllum commune',
    's_pombe': 'Schizosaccharomyces pombe',
    's_moellendorffii': 'Selaginella moellendorffii',
    's_viridis': 'Setaria viridis',
    's_lycopersicum': 'Solanum lycopersicum',
    's_purpuratus': 'Strongylocentrotus purpuratus',
    's_scrofa': 'Sus scrofa',
    't_guttata': 'Taeniopygia guttata',
    't_gondii': 'Toxoplasma gondii',
    't_castaneum': 'Tribolium castaneum',
    't_adhaerens': 'Trichoplax adhaerens',
    't_aestivum': 'Triticum aestivum',
    't_brucei': 'Trypanosoma brucei',
    'u_maydis': 'Ustilago maydis',
    'x_laevis': 'Xenopus laevis',
    'z_mays': 'Zea mays',
    'chordata': 'Chordata', 
    'nematoda': 'Nematoda', 
    'arthropoda': 'Arthropoda', 
    'placozoa': 'Placozoa', 
    'echinodermata': 'Echinodermata', 
    'apicomplexa': 'Apicomplexa', 
    'euglenozoa': 'Euglenozoa', 
    'evosea': 'Evosea', 
    'fornicata': 'Fornicata', 
    'streptophyta': 'Streptophyta', 
    'ascomycota': 'Ascomycota', 
    'basidiomycota': 'Basidiomycota', 
    'mucoromycota': 'Mucoromycota',
    'unknown': 'unknown'
}


def load_model(model_no):
    """
    Load the NetStart 2.0 model from Hugging Face Hub.

    Args:
        model_no (int [1, 2, 3, 4]): The model number to load.
    
    Returns:
        checkpoint (dict): The model checkpoint dictionary.
    """

    #Define the local directory where models will be cached
    local_dir = './data/data_model/models/'
    model_name = f'netstart_model{model_no}.pth'
    local_path = os.path.join(local_dir, model_name)
    
    #Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    #If model isn't downloaded yet, get it from Hugging Face
    if not os.path.exists(local_path):
        local_path = hf_hub_download(
            repo_id="linesandvad/netstart2_models", 
            filename=model_name,
            cache_dir=local_dir
        )
    
    #Load the model
    checkpoint = torch.load(local_path, map_location=torch.device('cpu'))
    
    return checkpoint


def validate_origin(origin):
    """
    Check if the origin is in the list of valid species or phyla.

    Args:
        origin (str): The species or phylum to validate.
    """

    if origin in conversion_dict.keys():
        return True, "Valid origin"
    return False, f"Invalid origin. Must be one of the valid species or phyla. Got: {origin}"


#Define functions for checking input format
def is_fasta_file(filename):
    """
    Check if a file is in FASTA format by verifying content format only.

    Args:
        filename (str): Path to the file to check.
    """

    #Check file existence
    if not os.path.exists(filename):
        return False, f"File {filename} does not exist"
    
    open_func = gzip.open if filename.endswith(".gz") else open
    
    #Check file content
    try:
        with open_func(filename, 'rt') as f:
            #Read first line
            first_line = f.readline().strip()
            if not first_line.startswith('>'):
                return False, "File is not in FASTA format: should start with '>'"
            
            #Check if file has sequence content
            has_sequence = False
            for line in f:
                if line.strip() and not line.startswith('>'):
                    has_sequence = True
                    break
            
            if not has_sequence:
                return False, "File contains no sequence data"
                
        return True, "Valid FASTA file"
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


#Define functions for processing input fasta file
def encode_nucleotide_to_amino_acid(sequence):
    """
    The function takes a nucleotide sequence and translates it into the corresponding amino acid sequence. 
    
    Args:
        sequence (str): A nucleotide sequence.
        
    Returns:
        amino_acid_sequence (str): A string representing the amino acid sequence translated from the 
                                   nucleotide input. Each codon is mapped to its corresponding amino acid,
                                   with stop codons represented by "<unk>".
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
        #Get amino acid (stop codons represented as unknown tokens, uncertain codons (with N etc.) represented as pad tokens)
        amino_acid = genetic_code.get(codon, "<pad>")
        amino_acid_sequence += amino_acid

    return amino_acid_sequence


def one_hot_encode(sequence):
    """
    One-hot encode nucleotide sequences in a matrix format of 4 rows (A, C, G, T)
    and len(sequence) columns.

    Args:
        sequence (str): a nucleotide sequence. 

    Returns: 
        encoding (tensor): one-hot encoded sequence. 
    """

    #Define the mapping of nucleotides to indices
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    #Create an empty one-hot encoding tensor
    encoding = torch.zeros(4, len(sequence))
    
    #Convert the sequence to a tensor of indices for efficient indexing
    indices = torch.tensor(
        [mapping[char] for char in sequence if char in mapping], dtype=torch.long)
    
    #Use advanced indexing to set the appropriate positions to 1
    positions = torch.arange(len(sequence))[[char in mapping for char in sequence]]
    encoding[indices, positions] = 1
        #For 'N', we do nothing, so the corresponding column remains all zeros
    
    return encoding


def load_taxonomy_mappings():
    """
    Load major taxonomy ranks (species, genus, family, order, class, phylum, kingdom)
    for each species in dataset.
    
    Returns:
        vocab_sizes (list): A list of integers representing the number of unique taxonomic groups at each rank. 
                            The list is ordered as [species, genus, family, order, class, phylum, kingdom].
        tax_mapping (dict): A dictionary where each key is a species name and each value is a list of string IDs 
                            corresponding to the taxonomic ranks [species, genus, family, order, class, phylum, kingdom].
                            Each ID is a unique numerical identifier for that rank, or an empty string if the rank is unavailable.
    
    
    """
    
    #Read in json files with taxonomy information
    with open("./data/data_model/taxonomy/species_names.json", 'r') as file:
        species_names_dict = json.load(file)
    with open("./data/data_model/taxonomy/species_to_kingdom.json", 'r') as file:
        species_kingdom_dict = json.load(file)
    with open("./data/data_model/taxonomy/species_to_phylum.json", 'r') as file:
        species_phylum_dict = json.load(file)
    with open("./data/data_model/taxonomy/species_to_class.json", 'r') as file:
        species_class_dict = json.load(file)
    with open("./data/data_model/taxonomy/species_to_order.json", 'r') as file:
        species_order_dict = json.load(file)
    with open("./data/data_model/taxonomy/species_to_family.json", 'r') as file:
        species_family_dict = json.load(file)
    with open("./data/data_model/taxonomy/species_to_genus.json", 'r') as file:
        species_genus_dict = json.load(file)

    #Create a list of unique IDs from all dictionaries, excluding '0' 
    #(missing value / species not classified at given rank)
    all_ids = set(species_names_dict.values()) | set(species_genus_dict.keys()) | set(species_family_dict.keys()) | set(species_order_dict.keys()) | set(species_class_dict.keys()) | set(species_phylum_dict.keys()) | set(species_kingdom_dict.keys())

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
            
    return vocab_sizes, tax_mapping


def read_fasta(filepath):
    """
    Generator function to read entries from a FASTA file (plain text or gzipped) one by one.
    
    Args:
        filepath (str): Path to the FASTA file.
        
    Yields:
        tuple: A tuple with (header, sequence) for each entry in the FASTA file.
    """

    #Determine if the file is gzipped or not and open accordingly
    open_func = gzip.open if filepath.endswith(".gz") else open
    
    #Open the file and read entries one by one
    with open_func(filepath, 'rt') as file:
        header = None
        sequence = []

        for line in file:
            line = line.strip()
            if line.startswith(">"): 
                 #Yield previous entry if exists
                if header: 
                    yield (header, ''.join(sequence))
                #Remove ">" from header
                header = line[1:]  
                #Reset sequence
                sequence = []
            #Accumulate sequence lines
            else:
                sequence.append(line)  

        #Yield the last entry after exiting loop
        if header:
            yield (header, ''.join(sequence))


def get_formatted_nucleotide_sequence(sequence, position, nucleotides_downstream, extract_upstream_nt, extract_downstream_nt):
    """
    Format nucleotide sequence to fit local input window (local start codon context) of NetStart 2.0. 

    Args: 
        sequence (str): the nucleotide sequence to process.
        position (int): The position of the targeted ATG in sequence.
        nucleotides_downstream (int): The number of nucleotides placed downstream of the targeted ATG in sequence.
        extract_upstream_nt (int): the number of nucleotides upstream of ATG to extract.
        extract_downstream_nt (int): the number of nucleotides downstream of ATG to extract.

    Returns: 
        sequence_nt (str): The formatted nucleotide subsequence.
    """
    
    sequence_nt = ""
                
    #If needed to extract more nucleotides than there is in sequence upstream
    if position < extract_upstream_nt: 
        pad_positions_nt_upstream = extract_upstream_nt - position
                    
        #add padding to start
        sequence_nt += pad_positions_nt_upstream*"N"
        sequence_nt += sequence[:position]
                    
    else:
        pad_positions_nt_upstream = 0
        sequence_nt += sequence[position-extract_upstream_nt:position]
                
    #If needed to extract more nucleotides than there is in sequence downstream
    if nucleotides_downstream < extract_downstream_nt: 
        sequence_nt += sequence[position:]
                    
        #Add padding to end
        pad_positions_nt_downstream = extract_downstream_nt - nucleotides_downstream
        sequence_nt += pad_positions_nt_downstream*"N"
                
    else:
        pad_positions_nt_downstream = 0
        sequence_nt += sequence[position:position+extract_downstream_nt+3]
        
    return sequence_nt


def get_formatted_amino_acid_sequence(sequence, position, nucleotides_upstream, nucleotides_downstream, extract_upstream_aa, extract_downstream_aa):
    """
    Translate nucleotide sequence to fit global input window (local start codon context) of NetStart 2.0. 

    Args: 
        sequence (str): the nucleotide sequence to process.
        position (int): The position of the targeted ATG in sequence.
        nucleotides_upstream (int): The position of the targeted ATG in sequence (modified if sequence length is not a multiplum of 3).
        nucleotides_downstream (int): The number of nucleotides placed downstream of the targeted ATG in sequence.
        extract_upstream_aa (int): the number of amino acids upstream of ATG to extract.
        extract_downstream_aa (int): the number of amino acids downstream of ATG to extract.

    Returns: 
        sequence_aa (str): The formatted amino acid subsequence.
    """
    
    #Pad sequences that are shorter than input sequence
    sequence_aa = ""
    pad_token = "<pad>"
    
    #If needed to extract more nucleotides than there is in sequence upstream
    if nucleotides_upstream < extract_upstream_aa * 3: 
        #If the number of nucleotides upstream is not a multiple of 3, round down to the nearest multiple of 3
        if nucleotides_upstream % 3 != 0:
            nucleotides_upstream = nucleotides_upstream - (nucleotides_upstream % 3)
        #Calculate the number of amino acids upstream
        aa_upstream = int(nucleotides_upstream / 3)
        pad_positions_aa = int(extract_upstream_aa - aa_upstream)
        sequence_aa += pad_positions_aa*pad_token
        sequence_aa += encode_nucleotide_to_amino_acid(sequence[position-nucleotides_upstream:position])
    else:
        sequence_aa += encode_nucleotide_to_amino_acid(sequence[position-extract_upstream_aa*3:position])
                
    #If needed to extract more nucleotides than there is in sequence downstream
    if nucleotides_downstream < extract_downstream_aa * 3:
        if nucleotides_downstream % 3 != 0:
            nucleotides_downstream = nucleotides_downstream - (nucleotides_downstream % 3)
                    
        aa_downstream = int(nucleotides_downstream / 3)
                    
        sequence_aa += encode_nucleotide_to_amino_acid(sequence[position:position+3+nucleotides_downstream])
        pad_positions_aa = int(extract_downstream_aa - aa_downstream)
        sequence_aa += pad_positions_aa*pad_token
    else:
        sequence_aa += encode_nucleotide_to_amino_acid(sequence[position:position+3+extract_downstream_aa*3])
        
    return sequence_aa

def extract_datasets(input_filename, 
                     extract_upstream_aa, 
                     extract_downstream_aa,
                     tax_mappings,
                     origin):
    """
    Get part of datasets that is the same across trained models in correct format to run NetStart 2.0 on. 
    
    Args:
        input_filename (str): the fasta input file.
        extract_upstream_aa (int): the number of amino acid positions upstream ATG to extract.
        extract_downstream_aa (int): the number of amino acid positions downstream ATG to extract.
        tax_mappings (dict): A dictionary containing taxonomic mappings for each species.
        origin (str): The species/phylum to format datasets for.
    
    Returns:
        df_input (df): dataframe with invariant sequence information required for model.
    """

    phylum_to_species_dict = {'Chordata': 'Homo sapiens', 
                          'Nematoda': 'Caenorhabditis elegans', 
                          'Arthropoda': 'Drosophila melanogaster', 
                          'Placozoa': 'Trichoplax adhaerens', 
                          'Echinodermata': 'Strongylocentrotus purpuratus', 
                          'Apicomplexa': 'Plasmodium falciparum', 
                          'Euglenozoa': 'Leishmania donovani', 
                          'Evosea': 'Dictyostelium discoideum', 
                          'Fornicata': 'Giardia intestinalis', 
                          'Streptophyta': 'Arabidopsis thaliana', 
                          'Ascomycota': 'Saccharomyces cerevisiae', 
                          'Basidiomycota': 'Cryptococcus neoformans', 
                          'Mucoromycota': 'Rhizophagus irregularis'}
    
    rows = []
    motif = "ATG"
    stop_codons = ["TAA", "TAG", "TGA"]
    
    #phylum-level representation          
    if origin in phylum_to_species_dict.keys(): 
        tax_mapping = ['0']*5 + tax_mappings[phylum_to_species_dict[origin]][-2:]
    #unknown origin
    elif origin == "unknown":
        tax_mapping = ['0']*7
    #species-level representation
    else: 
        tax_mapping = tax_mappings[origin]

    #Loop over entries in input fasta file
    for header, sequence in read_fasta(input_filename):
        sequence = sequence.upper().replace("U", "T")
        
        #Find all ATG positions in sequence
        positions = [match.start() for match in re.finditer(motif, sequence)]
        
        for position in positions:
            assert sequence[position:position+3] == motif
            
            #Find first in-frame stop codon after ATG
            position_1_indexed = position + 1
            first_stop_codon_pos_1_indexed = float('nan')
            aa_seq_len = float('nan')
            for i in range(position + 3, len(sequence), 3):
                codon = sequence[i:i+3]
                if codon in stop_codons:
                    first_stop_codon_pos_1_indexed = i + 1

                    nucleotide_seq_len = int(first_stop_codon_pos_1_indexed - position_1_indexed)
                    assert nucleotide_seq_len % 3 == 0, "Start- and stop codon positions not extracted properly."
                    aa_seq_len = nucleotide_seq_len // 3

                    break
            
            nucleotides_upstream = len(sequence[:position])
            nucleotides_downstream = len(sequence[position+3:])

            #Extract amino acid sequence          
            sequence_aa = get_formatted_amino_acid_sequence(
                sequence, 
                position, 
                nucleotides_upstream, 
                nucleotides_downstream, 
                extract_upstream_aa, 
                extract_downstream_aa
            )

            #Get all information shared between models required to do prediction for every sequence in fastafile.
            rows.append({
                'aa_sequences': sequence_aa,
                'tax_ranks': np.array(tax_mapping),
                'origin': origin,
                'entry_line': header,
                'atg_position': position_1_indexed,                     # ATG Position relative to full sequence (1-indexed)
                'stop_codon_position': first_stop_codon_pos_1_indexed,  # Stop Codon Position relative to full sequence (1-indexed) (position of first base in stop codon)
                'peptide_len': aa_seq_len
            })
            
    df_input = pd.DataFrame(rows)
    return df_input


def extract_datasets_nt(input_filename, 
                     extract_upstream_nt, 
                     extract_downstream_nt):
    """
    Get part of datasets that varies across trained models in correct format to run NetStart 2.0 on.
    
    Args:
        input_filename (str): the fasta input file.
        extract_upstream_nt (int): the number of nucleotide positions upstream ATG to extract.
        extract_downstream_nt (int): the number of nucleotide positions downstream ATG to extract.
    
    Returns: 
        df_input_nt (df): dataframe with formatted nucleotide sequences required for model.
    """
    
    rows = []
    motif = "ATG"
    
    for header, sequence in read_fasta(input_filename):
        
        sequence = sequence.upper().replace("U", "T")
        
        #Find all ATG positions in sequence
        positions = [match.start() for match in re.finditer(motif, sequence)]
        
        #loop over all ATG positions (or if specified, specific position(s))
        for position in positions:
            assert sequence[position:position+3] == motif
        
            #Extract the number of nucleotides upstream and downstream the ATG, respectively
            nucleotides_downstream = len(sequence[position+3:])
                      
            #####Extract nucleotide sequence####
            sequence_nt = get_formatted_nucleotide_sequence(sequence, position, nucleotides_downstream, extract_upstream_nt, extract_downstream_nt)

            #Get all information required to do prediction for every sequence in fastafile.    
            rows.append({
                'nt_sequences': sequence_nt})
            
    df_input_nt = pd.DataFrame(rows)
    return df_input_nt


def create_encodings_aa(df_input, extract_upstream_aa, extract_downstream_aa, batch_size=64):
    """
    Efficiently encode labels and sequences for pretrained models using batching.

    Args:
        df_input (dataframe): dataframe with input data
        extract_upstream_aa (int): Length of upstream amino acids to extract.
        extract_downstream_aa (int): Length of downstream amino acids to extract.
        batch_size (int): Size of each batch for tokenization.

    Returns:
        encodings_aa (dict): Dictionary containing tokenized input_ids and attention_masks.
    """

    #Compute the sequence length
    aa_seqs_len = extract_upstream_aa + 1 + extract_downstream_aa

    #Initialize the tokenizer
    tokenizer_aa = AutoTokenizer.from_pretrained(
        "facebook/esm2_t6_8M_UR50D",
        do_lower_case=False,
        model_max_length=aa_seqs_len + 2  # Include special tokens
    )

    #Split sequences into batches for efficiency
    sequences = df_input['aa_sequences'].tolist()
    encodings_aa = {'input_ids': [], 'attention_mask': []}

    for i in range(0, len(sequences), batch_size):
        #Process the current batch
        batch = sequences[i:i + batch_size]

        #Tokenize with padding and truncation
        batch_encodings = tokenizer_aa(
            batch,
            padding=True,  #Pad sequences to the max length in the batch
            truncation=True,
            return_tensors="pt"
        )

        #Manually create attention masks
        attention_masks = torch.ones_like(batch_encodings['input_ids'])
        pad_token_indices = batch_encodings['input_ids'] == tokenizer_aa.pad_token_id
        mask_token_indices = batch_encodings['input_ids'] == tokenizer_aa.mask_token_id
        attention_masks.masked_fill_(pad_token_indices | mask_token_indices, 0)

        #Append results to the encodings dictionary
        encodings_aa['input_ids'].append(batch_encodings['input_ids'])
        encodings_aa['attention_mask'].append(attention_masks)

    #Concatenate all batches into single tensors
    encodings_aa['input_ids'] = torch.cat(encodings_aa['input_ids'], dim=0)
    encodings_aa['attention_mask'] = torch.cat(encodings_aa['attention_mask'], dim=0)

    return encodings_aa


def create_encodings_nt(df_input):
    """
    Encode nucleotides sequences.

    Args:
        df_input (dataframe): dataframe with input data

    Returns: 
        sequences_nt (tensor): one-hot encoded nucleotide sequences
    """

    #Encode nucleotide sequences
    #One-hot encode and format all sequences in dataset
    sequences = [one_hot_encode(seq) for seq in df_input['nt_sequences']]
    sequences_nt = torch.stack(sequences)
    
    return sequences_nt


class InputDatasetNT(torch.utils.data.Dataset):
    """
    Create part of dataset to feed NetStart 2.0 that varies across trained models (PyTorch applicable format).

    Args:
        nt_encodings (tensor): the nucleotide encodings.
    """
    def __init__(self, nt_encodings):
        self.nt_encodings = nt_encodings

    def __getitem__(self, idx):
        """
        Returns the nucleotide encoding at the given index as a PyTorch tensor.
        """
        return {'nt_encodings': torch.as_tensor(self.nt_encodings[idx], dtype=torch.float32)}

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.nt_encodings)


class MultiInputDataset(torch.utils.data.Dataset): ##Write comments. 
    """
    Create part of dataset to feed NetStart 2.0 that varies across trained models (PyTorch applicable format).
    """
    def __init__(self, aa_encodings, tax_ranks, entry_line, atg_position, origin, stop_codon_position, peptide_len):
        self.aa_encodings = aa_encodings
        self.tax_ranks = tax_ranks
        self.entry_line = entry_line
        self.atg_position = atg_position
        self.origin = origin
        self.stop_codon_position = stop_codon_position
        self.peptide_len = peptide_len

    def __getitem__(self, idx):
        # Convert strings to integers in the 'tax_ranks' list
        tax_ranks_item = [int(rank) if rank != '0' else 0 for rank in self.tax_ranks[idx]]
        
        item = {
            'aa_encodings': {key: torch.as_tensor(val[idx]) for key, val in self.aa_encodings.items()},
            'tax_ranks': torch.LongTensor(tax_ranks_item),
            'entry_line': self.entry_line[idx],
            'atg_position': self.atg_position[idx],
            'origin': self.origin[idx],
            'stop_codon_position': self.stop_codon_position[idx],
            'peptide_len': self.peptide_len[idx]
        }
        return item

    def __len__(self):
        return len(self.origin)


class NetstartModel(nn.Module):
    """
    Define model architecture.
    """
    def __init__(self, 
                 aa_encoding_length, 
                 nt_encoding_length, 
                 vocab_sizes, 
                 emb_size,
                 hidden_neurons_aa,
                 hidden_neurons_nt,
                 hidden_neurons_merge,
                 dropout_rate_1,
                 dropout_rate_2,
                 model_no,
                 num_hidden_layers_nt,
                 num_labels = 1):
        
        super(NetstartModel, self).__init__()
        
        #Embedding size per taxonomical rank represented
        self.emb_size = emb_size

        ###Define layers###
        #Define pretrained model for amino acid encodings
        self.pretrained_model_aa = AutoModel.from_pretrained("./data/data_model/pretrained_models/finetuned_models/esm2-8m-finetuned_model_100u_100d_model"+str(model_no))
        
        #Define hidden layer to downscale amino acid sequence representations
        self.hidden_layer_aa = nn.Linear(self.pretrained_model_aa.config.hidden_size*aa_encoding_length, hidden_neurons_aa)

        #Define feedforward hidden layers for local start codon context window
        self.nt_layers = nn.ModuleList()
        self.nt_layers.append(nn.Linear(nt_encoding_length, hidden_neurons_nt))  # First layer

        #Additional hidden layers for nt encoding if `num_hidden_layers_nt > 1`
        for _ in range(num_hidden_layers_nt - 1):
            self.nt_layers.append(nn.Linear(hidden_neurons_nt, hidden_neurons_nt))
        
        #Define embeddings for the major taxonomical ranks
        self.embedding_species = nn.Embedding(vocab_sizes[0], emb_size, padding_idx=0)
        self.embedding_genus = nn.Embedding(vocab_sizes[1], emb_size, padding_idx=0)
        self.embedding_family = nn.Embedding(vocab_sizes[2], emb_size, padding_idx=0)
        self.embedding_order = nn.Embedding(vocab_sizes[3], emb_size, padding_idx=0)
        self.embedding_class = nn.Embedding(vocab_sizes[4], emb_size, padding_idx=0)
        self.embedding_phylum = nn.Embedding(vocab_sizes[5], emb_size, padding_idx=0)
        self.embedding_kingdom = nn.Embedding(vocab_sizes[6], emb_size, padding_idx=0)
        
        #Define hidden layer to merge all individual windows
        self.hidden_layer_2 = nn.Linear(hidden_neurons_aa+hidden_neurons_nt+emb_size, hidden_neurons_merge)

        #Define output layer
        self.classifier = nn.Linear(hidden_neurons_merge, num_labels)
        
        #Define dropout
        self.dropout_1 = nn.Dropout(dropout_rate_1)
        self.dropout_2 = nn.Dropout(dropout_rate_2)

        #Define number of labels
        self.num_labels = num_labels
        
        #Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, 
                x_aa, 
                x_nt, 
                attention_mask_aa, 
                tax_ranks):
        
        """ 
        Forward pass through the model.

        Args:
            x_aa (tensor): amino acid encodings.
            x_nt (tensor): nucleotide encodings.
            attention_mask_aa (tensor): attention mask for amino acid encodings.
            tax_ranks (tensor): taxonomical ranks.
        """
        
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
        output = self.sigmoid(output)
        
        return output
    

def ExtractDataAndModel(model_no, input_filename, aa_encodings_len, vocab_sizes, batch_size):
    """
    Extract data in correct format and instantiate model with specified, optimized hyperparameters.

    Args: 
        model_no (int [1, 2, 3, 4]): the model number
        input_filename (str): path to input fasta file
        aa_encodings_len (int): The length of amino acid encodings input to model
        vocab_sizes (list): vocab size for each taxonomical level

    Returns: 
        model: model checkpoint (correspondng to specific trained model)
        dataloader_nt: nucleotide input sequences prepared for input to model 
    """

    #Load the json file back into Python as a dictionary
    with open("./data/data_model/hyperparameters/netstart_model"+str(model_no)+"_hyperparameters.json", 'r') as json_file:
        model_config = json.load(json_file)
    
    #Get processed sequences and required information for model
    df_input_nt = extract_datasets_nt(input_filename,
                                      extract_upstream_nt = model_config["nt_upstream"], 
                                      extract_downstream_nt = model_config["nt_downstream"])
    
    #Create nucleotide encodings
    encodings_nt = create_encodings_nt(df_input_nt)
    
    nt_encodings_len_flattened = len(encodings_nt[0])*len(encodings_nt[0][0]) #4 * numbers of nt

    dataset_nt = InputDatasetNT(encodings_nt)
                                
    dataloader_nt = DataLoader(dataset_nt, batch_size=batch_size, shuffle=False)
    
    checkpoint = load_model(model_no)

    #Remove 'module.' prefix if present and assigning the entire loaded dictionary as the state dictionary
    model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    #Create an instance of your model with the required parameters
    #Provide the necessary parameters used during model creation
    model = NetstartModel(aa_encodings_len, 
                          nt_encodings_len_flattened, 
                          vocab_sizes, 
                          model_config["emb_size_tax"], 
                          model_config["hidden_neurons_aa"], 
                          model_config["hidden_neurons_nt"], 
                          model_config["hidden_neurons_merge"], 
                          model_config["dropout_rate_1"], 
                          model_config["dropout_rate_2"],
                          model_no = model_no,
                          num_hidden_layers_nt=model_config["depth_nt_window"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #Instantiate model with checkpoint
    model.load_state_dict(model_state_dict)

    #Set the model to evaluation mode
    model.eval()
    
    return model, dataloader_nt


def get_predictions(origin, input_filename, output_filename, output_results, threshold, gzip_outfile, batch_size):
    """
    Run entire pipeline to get predictions.

    Args: 
        origin (str): The name of the species/phylum to get predictions for.
        input_filename (str): path to and filename of input fasta file (can be either fasta or fasta.gz).
        output_filename (str): the output filename prefix. 
        out_format (str) ["csv", "json", "both"]: the format the predictions should be outputted in (default: "both"). 
        threshold (float): the threshold to use for binary classification (default: 0.5).
        gzip_outfile (bool): whether to gzip the output files (default: False).
    
    """
    print("Preparing data for model input.")
    start_time = time.time()
    vocab_sizes, tax_mappings = load_taxonomy_mappings()
    origin = conversion_dict[origin]

    df_input = extract_datasets(input_filename,
                                extract_upstream_aa = 100,  
                                extract_downstream_aa = 100,
                                tax_mappings = tax_mappings,
                                origin = origin)

    #Check if there are any ATGs to predict on
    if len(df_input) > 0:
        
        #Create amino acid encodings
        encodings_aa = create_encodings_aa(df_input, 100, 100)
        aa_encodings_len = encodings_aa['input_ids'].shape[1]

        #Process dataset to fit as model input
        dataset_invariant = MultiInputDataset(encodings_aa,
                                        df_input['tax_ranks'], 
                                        df_input['entry_line'],
                                        df_input['atg_position'],
                                        df_input['origin'],
                                        df_input['stop_codon_position'],
                                        df_input['peptide_len'])
        
        #Create dataloader for data not varying across models
        test_loader_invariant = DataLoader(dataset_invariant, batch_size=batch_size, shuffle=False)

        #Get varying data and models 
        model1, test_loader1 = ExtractDataAndModel(model_no = "1", 
                                                input_filename = input_filename, 
                                                aa_encodings_len = aa_encodings_len, 
                                                vocab_sizes = vocab_sizes,
                                                batch_size = batch_size)
        model2, test_loader2 = ExtractDataAndModel(model_no = "2", 
                                                input_filename = input_filename, 
                                                aa_encodings_len = aa_encodings_len, 
                                                vocab_sizes = vocab_sizes,
                                                batch_size = batch_size)
        model3, test_loader3 = ExtractDataAndModel(model_no = "3", 
                                                input_filename = input_filename, 
                                                aa_encodings_len = aa_encodings_len, 
                                                vocab_sizes = vocab_sizes,
                                                batch_size = batch_size)
        model4, test_loader4 = ExtractDataAndModel(model_no = "4", 
                                                input_filename = input_filename, 
                                                aa_encodings_len = aa_encodings_len, 
                                                vocab_sizes = vocab_sizes,
                                                batch_size = batch_size)
        
        #Calculate and print total time
        total_time = time.time() - start_time
        print(f"\nData processing finished. Processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes).")
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Initialize
        information_dict = dict()
        information_dict["origin"] = []
        information_dict["atg_position"] = []
        information_dict["stop_codon_position"] = []
        information_dict['peptide_len'] = []
        information_dict["entry_line"] = []
        information_dict["preds"] = []

        assert len(test_loader1) == len(test_loader2) == len(test_loader3) == len(test_loader4) == len(test_loader_invariant), "Data not loaded properly."

        total_batches = len(test_loader1)
        
        #Predict on all batches
        with torch.no_grad():
            for batch_1, batch_2, batch_3, batch_4, batch_invariant in tqdm(zip(test_loader1, test_loader2, test_loader3, test_loader4, test_loader_invariant), 
                                                            total=total_batches, desc="Predicting"):
                
                #Get the input data being invariant for all 4 models
                inputs_aa = batch_invariant["aa_encodings"]["input_ids"].to(device)
                attention_mask_aa = batch_invariant["aa_encodings"]["attention_mask"].to(device)
                tax_ranks = batch_invariant["tax_ranks"].to(device)

                #Get data specific to each model (nucleotide input window)
                inputs_nt_1 = batch_1["nt_encodings"].to(device)
                inputs_nt_2 = batch_2["nt_encodings"].to(device)
                inputs_nt_3 = batch_3["nt_encodings"].to(device)
                inputs_nt_4 = batch_4["nt_encodings"].to(device)

                #Get predictions and loss for each batch
                outputs_1 = model1(inputs_aa, inputs_nt_1, attention_mask_aa, tax_ranks)
                outputs_2 = model2(inputs_aa, inputs_nt_2, attention_mask_aa, tax_ranks)
                outputs_3 = model3(inputs_aa, inputs_nt_3, attention_mask_aa, tax_ranks)
                outputs_4 = model4(inputs_aa, inputs_nt_4, attention_mask_aa, tax_ranks)

                outputs_averaged = (outputs_1+outputs_2+outputs_3+outputs_4)/4

                information_dict["preds"].extend(outputs_averaged.cpu().numpy().flatten())
                information_dict["origin"].extend(batch_invariant["origin"])
                information_dict["atg_position"].extend(batch_invariant["atg_position"])
                information_dict["stop_codon_position"].extend(batch_invariant["stop_codon_position"])
                information_dict['peptide_len'].extend(batch_invariant['peptide_len'])
                information_dict["entry_line"].extend(batch_invariant["entry_line"])
                
        #Convert float32 values to native Python floats in the dictionary lists
        information_dict["preds"] = [round(float(val), 6) for val in information_dict["preds"]]
        information_dict["atg_position"] = [int(val) for val in information_dict["atg_position"]]
        information_dict["stop_codon_position"] = [int(val.item()) if not torch.isnan(val) else float('nan') for val in information_dict["stop_codon_position"]]
        information_dict['peptide_len'] = [int(val.item()) if not torch.isnan(val) else float('nan') for val in information_dict['peptide_len']]

        information_df = pd.DataFrame(information_dict)

        if output_results == "max_prob":
            # Use groupby and idxmax to find the rows with the highest preds for each entry_line
            information_df = information_df.loc[information_df.groupby("entry_line")["preds"].idxmax()]
        
        elif output_results == "threshold":
            information_df = information_df[information_df["preds"] > threshold]

        if gzip_outfile:
            information_df.to_csv(output_filename+".csv.gz", index=False, compression='gzip')
        else:
            information_df.to_csv(output_filename+".csv", index=False)

    else:
        return None
    


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Predict Eukaryotic Translation Initiation Sites with NetStart 2.0')
    
    # Add the arguments
    parser.add_argument('-o', '--origin', 
                       type=str,
                       required=True,
                       help='Input the origin of the sequence.')
    
    parser.add_argument('-in', '--input_filename',
                       type=str,
                       required=True,
                       help='Input file name in FASTA format (can also be in gzipped format with .gz-extension).')
    
    parser.add_argument('-out', '--output_filename',
                       type=str,
                       default="netstart2_preds_out",
                       help='Output file name without file extension.')
    
    parser.add_argument('--output_results',
                       type=str,
                       default="all",
                       help='Your wanted output. Choose between "max_prob", "threshold" or "all" (default: all).')
    
    parser.add_argument('--threshold',
                        type=float,
                        default=0.625,
                        help='Set the threshold for filtering predictions. Only works with "--output_results threshold" (default: 0.625).')

    parser.add_argument('--gzip_outfile', 
                        action='store_true',   
                        default=False,
                        help='Specify if output file should be gzipped (default: False).')
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Set batch size (default: 64).')

    # Parse the arguments
    args = parser.parse_args()

    # Validate origin
    is_valid_origin, origin_message = validate_origin(args.origin)
    if not is_valid_origin:
        print(f"Error: {origin_message}")
        sys.exit(1)
    
    # Validate FASTA file
    is_valid_fasta, fasta_message = is_fasta_file(args.input_filename)
    if not is_valid_fasta:
        print(f"Error: {fasta_message}")
        sys.exit(1)
    
    # Call the prediction function with parsed arguments
    get_predictions(origin = args.origin, 
                    input_filename = args.input_filename, 
                    output_filename = args.output_filename, 
                    output_results = args.output_results, 
                    threshold = args.threshold,
                    gzip_outfile = args.gzip_outfile,
                    batch_size=args.batch_size)


if __name__ == "__main__":
    main()