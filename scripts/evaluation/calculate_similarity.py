import numpy as np
import gzip
import h5py
from multiprocessing import Pool, cpu_count


def calculate_identity(seq1, seq2):
    """
    Calculate the sequence identity between two sequences.

    Args:
        seq1 (str): First sequence.
        seq2 (str): Second sequence.
    
        Returns:
            float: The sequence identity as a percentage.
    """
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / len(seq1) * 100

def align_sequences_at_tis(seq1, tis_pos1, seq2, tis_pos2):
    """
    Aligns two sequences at the same TIS position, ensuring both sequences 
    have the same length by trimming upstream and downstream sequences.

    Args:
        seq1 (str): First sequence. 
        tis_pos1 (int): TIS position for the first sequence.
        seq2 (str): Second sequence.
        tis_pos2 (int): TIS position for the second sequence.

    Returns:
        aligned_seq1 (str): First sequence aligned to the TIS position.
        aligned_seq2 (str): Second sequence aligned to the TIS position.
        aligned_tis_pos (int): TIS position for both sequences after alignment.
    """
    
    # Calculate upstream and downstream lengths for both sequences
    upstream1 = tis_pos1
    downstream1 = len(seq1) - tis_pos1 - 1
    
    upstream2 = tis_pos2
    downstream2 = len(seq2) - tis_pos2 - 1
    
    # Determine the minimum available upstream and downstream lengths
    upstream_min = min(upstream1, upstream2)
    downstream_min = min(downstream1, downstream2)
    
    # Align both sequences to the same TIS position
    # Slice the sequences to ensure equal upstream and downstream lengths
    aligned_seq1 = seq1[tis_pos1 - upstream_min : tis_pos1 + downstream_min + 1]
    aligned_seq2 = seq2[tis_pos2 - upstream_min : tis_pos2 + downstream_min + 1]
    
    # Both sequences should have the same TIS position in the aligned sequences
    aligned_tis_pos = upstream_min  # The TIS position for both sequences after alignment
    
    return aligned_seq1, aligned_seq2, aligned_tis_pos

#Conversion table to get transcript sequence from TIS Transformer
conversion_table = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}

transcript_seqs_TIS_transformer = {}

#Set of contigs to remove (TIS Transformer testset)
contigs_to_remove = {b'13', b'19', b'1', b'7'}

print("Extracting training sequences from TIS Transformer")
# Open the H5 file in read mode
with h5py.File("../../data/data_evaluation/TIS_transformer/dataset/GRCh38_v107.h5", 'r') as h5_file:
    # List all groups in the file
    print("Data:", list(h5_file.keys()))
    
    # Access a specific group
    group = h5_file['transcript']  # Replace 'group_name' with an actual group in your file
    print("Transcript contents:", list(group.keys()))
    
    #Access a dataset
    contigs = group['contig']
    ids = group["id"]
    metadata = group["metadata"]
    seq = group["seq"]
    tis_labels = group["tis"]

    # Loop through all transcripts
    for i in range(len(tis_labels)):
        contig = contigs[i]  # Get the contig for the current transcript
        
        #Skip the entry if the contig is in the removal list
        if contig in contigs_to_remove:
            continue
        
        #Get all entries labelled non-zero
        tis_pos = np.nonzero(tis_labels[i])[0]

        #If transcript is coding
        if len(tis_pos) > 0:
            assert len(tis_pos) == 1  # Ensure there's only one TIS position
            
            # Get only ATG TIS transcripts
            if seq[i][tis_pos] == 0 and seq[i][tis_pos + 1] == 1 and seq[i][tis_pos + 2] == 3:
                # Store the relevant information in the dictionary
                transcript_seqs_TIS_transformer[ids[i]] = {}
                
                # Convert sequence to nucleotide string
                nucleotide_sequence = ''.join(conversion_table[num] for num in seq[i])
                transcript_seqs_TIS_transformer[ids[i]]["tis_pos"] = tis_pos[0]  #Store the first TIS position
                transcript_seqs_TIS_transformer[ids[i]]["seq"] = nucleotide_sequence


#Load test partition
species_list = ['alligator_mississippiensis', 'anolis_carolinensis', 'anopheles_gambiae', 'apis_mellifera', 'arabidopsis_thaliana', 'aspergillus_nidulans', 'bos_taurus', 'brachypodium_distachyon', 'caenorhabditis_elegans', 'canis_lupus_familiaris', 'columba_livia', 'coprinopsis_cinerea', 'cryptococcus_neoformans', 'danio_rerio', 'daphnia_carinata', 'dictyostelium_discoideum', 'drosophila_melanogaster', 'eimeria_maxima', 'entamoeba_histolytica', 'equus_caballus', 'gallus_gallus', 'giardia_intestinalis', 'glycine_max', 'gorilla_gorilla', 'homo_sapiens', 'hordeum_vulgare', 'leishmania_donovani', 'lotus_japonicus', 'manduca_sexta', 'medicago_truncatula', 'mus_musculus', 'neurospora_crassa', 'nicotiana_tabacum', 'oreochromis_niloticus', 'oryctolagus_cuniculus', 'oryza_sativa', 'oryzias_latipes', 'ovis_aries', 'pan_troglodytes', 'phoenix_dactylifera', 'plasmodium_falciparum', 'rattus_norvegicus', 'rhizophagus_irregularis', 'saccharomyces_cerevisiae', 'schizophyllum_commune', 'schizosaccharomyces_pombe', 'selaginella_moellendorffii', 'setaria_viridis', 'solanum_lycopersicum', 'strongylocentrotus_purpuratus', 'sus_scrofa', 'taeniopygia_guttata', 'toxoplasma_gondii', 'tribolium_castaneum', 'trichoplax_adhaerens', 'triticum_aestivum', 'trypanosoma_brucei', 'ustilago_maydis', 'xenopus_laevis', 'zea_mays']

def process_species(species):
    """ 
    Processes the test data for a given species, calculates sequence similarities, and writes the results to a file.

    Args:
        species (str): The name of the species to process.
    """
    print(f"Loading {species} test data", flush=True)
    transcript_seqs_testset = {}
    write_seq = False

    # Load and process test set data
    with gzip.open(f"../../data/data_evaluation/input_testsets/mRNA_testsets_processed/input_testset_{species}_softmasked.fasta.gz", "rt") as testset_homo_sapiens:
        for line in testset_homo_sapiens:
            if line.startswith(">"):
                if "TIS=1" in line:
                    write_seq = True
                    tis_pos = line.split("ATG_pos=")[1].split("|")[0]
                    seq_number = line.split("seq_number=")[1].split("|")[0]
            else:
                if write_seq:
                    transcript_seqs_testset[seq_number] = {}
                    transcript_seqs_testset[seq_number]["tis_pos"] = int(tis_pos)
                    transcript_seqs_testset[seq_number]["seq"] = line.strip()
                write_seq = False

    similar_seqs_file = open(f"./similar_sequences_{species}.txt", "w")
    counter = 0

    print("Calculating similarities")
    for transcript_id_testset, data in transcript_seqs_testset.items():
        sequence_testset = data.get("seq").upper()
        tis_pos_testset = int(data.get("tis_pos"))

        similarity = 0

        for transcript_id, data in transcript_seqs_TIS_transformer.items():
            sequence_TIS_transformer = data.get("seq")
            tis_pos_TIS_transformer = int(data.get("tis_pos"))

            aligned_seq_testset, aligned_seq_TIS_transformer, aligned_tis_pos = align_sequences_at_tis(sequence_testset, tis_pos_testset, sequence_TIS_transformer, tis_pos_TIS_transformer)

            assert len(aligned_seq_testset) == len(aligned_seq_TIS_transformer)
            assert aligned_seq_testset[aligned_tis_pos:aligned_tis_pos+3] == aligned_seq_TIS_transformer[aligned_tis_pos:aligned_tis_pos+3] == "ATG"

            seq_identity = calculate_identity(aligned_seq_testset, aligned_seq_TIS_transformer)

            if seq_identity > similarity:
                similarity = seq_identity

        print(similarity, transcript_id_testset)
        similar_seqs_file.write(transcript_id_testset+"|"+str(similarity)+"\n")

        counter += 1

        if counter % 50 == 0:
            print(f"Processed {counter} transcripts.")

    similar_seqs_file.close()

    
# Parallel execution
if __name__ == "__main__":
    num_cpus = cpu_count()
    print(f"Using {num_cpus} CPUs")
    with Pool(num_cpus) as pool:
        pool.map(process_species, species_list)
