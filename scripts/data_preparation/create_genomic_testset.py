import gzip
import os
import pandas as pd
from Bio import SeqIO

def get_test_sequences_TIS():
	"""
	Reads a compressed CSV file containing graph partition assignments, filters the sequences assigned to the test partition (cluster 4), and organizes
	them into a dictionary where the keys are species names and the values are lists of sequence numbers.

	Returns:
		TIS_test_seqs_dict (dict): A dictionary with species names as keys and lists of test sequence numbers as values.
	"""

	#Initialize
	TIS_test_seqs_dict = {}

	#Open file with graphpart assignments
	with gzip.open(f"../../data/data_model_preparation/graphpart_assignments/graphpart_assignments_TIS.csv.gz", "rt") as file:
		df_TIS_partitions = pd.read_csv(file)  
	
	#Get information from test partition (4)
	TIS_testset = df_TIS_partitions[df_TIS_partitions["cluster"] == 4.0]
	TIS_test_seqs_list = list(TIS_testset["AC"])

	#Fill dict with species names as keys, test sequence numbers as values
	for seq_info in TIS_test_seqs_list:
		str_parts = seq_info.split('_')
		species_name = "_".join(str_parts[3:])
		seq_number = "_".join(str_parts[:3])

		if species_name not in TIS_test_seqs_dict:
			TIS_test_seqs_dict[species_name] = []

		TIS_test_seqs_dict[species_name].append(seq_number)

	return TIS_test_seqs_dict


def get_species_names():
	"""
	Extract a list of all species names.

	Returns:
		species_list (list): A list of all species names.
	"""

	species_filenames_list = os.listdir("../../data/data_raw/genomic_gff")
	species_list = [filename.split("_genomic.gff.gz")[0] for filename in species_filenames_list]

	return species_list

def load_gff_file(species):
	"""
	Load in gff-formatted annotations for specified species.

	Args:
		species (str): The species name.
	
	Returns:
		species_gff_df (DataFrame): gff annotations for the specified species.
	"""
	
	with gzip.open(f"../../data/data_raw/genomic_gff/{species}_genomic.gff.gz", 'rt') as file:
		species_gff_df = pd.read_csv(file, sep='\t', comment="#", header=None)

	return species_gff_df

def load_TIS_dataset(species):
	"""
	Load in the TIS-labelled dataset for specified species.

	Args:
		species (str): The species name.
	
	Returns:
		species_TIS_df (DataFrame): TIS-labelled dataset for the specified species.
	"""
	with gzip.open(f"../../data/data_model_preparation/datasets/TIS/mRNA_positive_{species}.csv.gz", "rt") as file:
		species_TIS_df = pd.read_csv(file) 

	return species_TIS_df

def read_fasta(species):
	"""
	Read in fasta file, stores one landmark sequence at a time.

	Args:
		species (str): The species name.
	
	Yields:	
		entry_line (str): The entry line of the fasta file.
		sequence (str): The sequence of the fasta file.
	"""
	with gzip.open(f"../../data/data_raw/genomic_fna/{species}_genomic.fna.gz", 'rt') as file:
		entry_line = None
		sequence = []
        
		for line in file:
			line = line.strip()
            
            #If it's an entry line line (starts with ">")
			if line.startswith(">"):
                #If we already have an entry line and sequence, yield them
				if entry_line is not None:
					yield entry_line, ''.join(sequence)
                
                #Start a new sequence with the new entry line
				entry_line = line
				sequence = []
			else:
                #Otherwise, it's part of the sequence; accumulate it
				sequence.append(line)
        
        #After the loop ends, yield the last sequence
		if entry_line is not None:
			yield entry_line, ''.join(sequence)

def reverse_complement(seq): 
	"""
	Take the reverse complement of a sequence for mRNAs located on the complement strand.

	Args:
		seq (str): The sequence to take reversed complement of. 
	
	Returns:
		reversed_complement (str): The reverse complement of the input sequence.
	"""
		
	#Map basepairs
	complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
					  'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n',
					  'R': 'N', 'K': 'N', 'Y': 'N', 'S': 'N', 'W': 'N', 
					  'r': 'n', 'k': 'n', 'y': 'n', 's': 'n', 'w': 'n',
					  'M': 'N', 'm': 'n', 'D': 'N', 'B': 'N'}
		
	#Get complement sequence
	complemented_sequence = [complement_map[base] for base in seq]

	#Reverse the complemented sequence
	reversed_complement = ''.join(complemented_sequence[::-1])

	return reversed_complement

def extract_genomic_testset(species, TIS_test_seqs_dict, genomic_testsets_dir, extend_gene_end_upstream, extend_gene_end_downstream):
	"""
	Create testset for TIS-labelled sequences on genomic level. 

	Args:
		species (str): The species name.
		TIS_test_seqs_dict (dict): Dictionary with species names as keys, sequence numbers of TIS-labelled sequences in test partition as values. 
		genomic_testsets_dir (str): The directory to store the genomic test sets.
		extend_gene_end_upstream (int): The number of nucleotides to extend upstream of the gene.
		extend_gene_end_downstream (int): The number of nucleotides to extend downstream of the gene.

	Output:
		genomic_testset_{species}.fna.gz (file): A compressed fasta file containing the genomic test set sequences.
	"""

	#Initialize 
	seqs_counter = 0

	#Open file to write to
	outfile_test_sequences = gzip.open(f"../../data/data_evaluation/input_testsets/genomic_testsets/{genomic_testsets_dir}/genomic_testset_{species}.fna.gz", "wt")

	#Load relevant data
	species_gff_df = load_gff_file(species)
	species_TIS_df = load_TIS_dataset(species)
		
	#Extract information on all test sequences from species
	test_seqs_list = list(TIS_test_seqs_dict[species])
	test_seqs_no = len(test_seqs_list)
	test_sequences_data_df = species_TIS_df[species_TIS_df["Seq_number"].isin(test_seqs_list)]

	print("Extracting on ", len(test_seqs_list), "test sequences.")

	#Loop over every landmark sequence in genomic fasta
	for entry_line, sequence in read_fasta(species):
		#Get landmark ID
		landmark = entry_line.split(" ")[0].split(">")[1]

		#Extract all test sequences on that landmark
		landmark_sequences_df = test_sequences_data_df[test_sequences_data_df["landmark_id"] == landmark]

		#Loop over all datapoints in test partition
		for index, row in landmark_sequences_df.iterrows():
			#Initial check that ATGs are placed correctly for each transcript
			if row.strand == "+":
				atg_coords = row.ATG_genomic_position[1:-1].split(", ")
				assert sequence[int(atg_coords[0]):int(atg_coords[1])].upper() == "ATG", "ATG not found on correct position!"
			elif row.strand == "-":
				atg_coords = row.ATG_genomic_position[1:-1].split(", ")
				assert reverse_complement(sequence[int(atg_coords[0]):int(atg_coords[1])].upper()) == "ATG", "ATG not found on correct position!"

			#Get all gff-annotations of corresponding transcript ID (mRNA, exon, CDS-annotations)
			mrna_annotations_df = species_gff_df[species_gff_df[8].str.contains("rna-"+row["transcript_id"]+";", na=False)]
			gene_annotations_df = species_gff_df[species_gff_df[8].str.contains(f"ID=gene-{row['gene']};", regex=False, na=False)]

			#Get first CDS annotation
			first_CDS_row = mrna_annotations_df[mrna_annotations_df[2] == "CDS"].iloc[0]

			#Get gene and mRNA annotation and annotation source
			mrna_row = mrna_annotations_df[mrna_annotations_df[2] == "mRNA"].iloc[0]
			gene_row = gene_annotations_df[gene_annotations_df[2] == "gene"].iloc[0]
			annotation_source =	gene_row[1]

			#Processed sequence counter
			seqs_counter += 1

			flanking_seq_missing_5 = "False"
			flanking_seq_missing_3 = "False"
			TSS_annotated = "True"

			#If transcript annotations are on template strand
			if first_CDS_row[6] == "+":

				#Transcription start site missing
				if mrna_row[3] == first_CDS_row[3]:
					TSS_annotated = "False"
					extend_gene_end_upstream += 180 #estimated 5' UTR length

				#Start position of CDS (TIS)
				start_CDS = first_CDS_row[3]

				#Find end position of first exon in coding sequence and start position of first downstream intron
				exon_end = mrna_annotations_df.loc[(mrna_annotations_df[2] == 'exon') & (mrna_annotations_df[4] > start_CDS), 4].min()  # End of the last exon before CDS
				first_downstream_intron_position = exon_end - start_CDS + 1 #if number is 16, then intron starts at position 17

				##extract gene sequence + additional sequence chunk upstream and downstream##
				#If the gene is placed in the very beginning of landmark (number of nucleotides upstream gene is less than 300)
				if gene_row[3] <= extend_gene_end_upstream and gene_row[4] <= len(sequence) - extend_gene_end_downstream:
					print("Identified on + strand: too short 5' end of landmark.")
					gene_seq = sequence[0:gene_row[4] + extend_gene_end_downstream]
					TIS_pos = start_CDS - 1
					flanking_seq_missing_5 = "True"


				#If the gene is placed in the very end of landmark (number of nucleotides downstream gene is less than 300)
				elif gene_row[4] >= len(sequence) - extend_gene_end_downstream and gene_row[3] >= extend_gene_end_upstream:
					print("Identified on + strand: too short 3' end of landmark.")
					gene_seq = sequence[gene_row[3] - 1 - extend_gene_end_upstream:]
					TIS_pos = start_CDS - gene_row[3] + extend_gene_end_upstream
					flanking_seq_missing_3 = "True"

				elif gene_row[3] <= extend_gene_end_upstream and gene_row[4] >= len(sequence) - extend_gene_end_downstream:
					print("Identified on + strand: too short 5' and 3' end of landmark.")
					gene_seq = sequence
					TIS_pos = start_CDS - 1
					flanking_seq_missing_5 = "True"
					flanking_seq_missing_3 = "True"
				
				#When possible to extract full gene sequence + 300 nucleotides both upstream and downstream
				else:
					gene_seq = sequence[gene_row[3] - 1 - extend_gene_end_upstream:gene_row[4] + extend_gene_end_downstream]
					TIS_pos = start_CDS - gene_row[3] + extend_gene_end_upstream

			#If transcript annotations are on complement strand
			elif first_CDS_row[6] == "-":

				#Transcription start site missing
				if mrna_row[4] == first_CDS_row[4]:
					TSS_annotated = "False"
					extend_gene_end_upstream += 180 #estimated 5' UTR length

				#Start position of CDS (TIS)
				start_CDS = first_CDS_row[4]

				#Find end position of first exon in coding sequence and start position of first downstream intron
				exon_end = mrna_annotations_df.loc[(mrna_annotations_df[2] == 'exon') & (mrna_annotations_df[3] < start_CDS), 3].max()  # End of the last exon before CDS
				first_downstream_intron_position = start_CDS - exon_end + 1 #if number is 8, then intron starts at position 9

				##extract gene sequence + additional sequence chunk upstream and downstream##
				#If the gene is placed in the very beginning of landmark (number of nucleotides upstream gene is less than 300)
				if gene_row[3] <= extend_gene_end_downstream and len(sequence) - int(gene_row[4]) >= extend_gene_end_upstream:
					print("Identified on - strand: too short 3' end of landmark.")
					gene_seq = reverse_complement(sequence[0:gene_row[4] + extend_gene_end_upstream])
					TIS_pos = gene_row[4] - start_CDS + extend_gene_end_upstream
					flanking_seq_missing_3 = "True"

				#If the gene is placed in the very end of landmark (number of nucleotides downstream gene is less than 300)
				elif len(sequence) - int(gene_row[4]) <= extend_gene_end_upstream and gene_row[3] >= extend_gene_end_downstream:

					print("Identified on - strand: too short 5' end of landmark.")
					gene_seq = reverse_complement(sequence[gene_row[3] - 1 - extend_gene_end_downstream:])
					TIS_pos = len(sequence) - start_CDS
					flanking_seq_missing_5 = "True"

				elif len(sequence) - int(gene_row[4]) <= extend_gene_end_upstream and gene_row[3] <= extend_gene_end_downstream:
					print("Identified on - strand: too short 5' and 3' end of landmark.")
					gene_seq = reverse_complement(sequence)
					TIS_pos = len(sequence) - start_CDS
					flanking_seq_missing_5 = "True"
					flanking_seq_missing_3 = "True"

				
				#When possible to extract full gene sequence + 300 nucleotides both upstream and downstream
				else:
					gene_seq = reverse_complement(sequence[gene_row[3] - 1 - extend_gene_end_downstream:gene_row[4] + extend_gene_end_upstream])
					TIS_pos = gene_row[4] - start_CDS + extend_gene_end_upstream

			if TSS_annotated == "False":
				extend_gene_end_upstream -= 180

			assert extend_gene_end_upstream == 2000, "Upstream region extracted improperly."

			#Ensure that TIS positions are correct
			assert gene_seq[TIS_pos:TIS_pos+3].upper() == "ATG", print(gene_seq, TIS_pos, gene_annotations_df, mrna_annotations_df)

			gene_name = row["gene"].replace(' ', '')
			
			#Write entry line and gene sequence to outfile
			outfile_test_sequences.write(">"+gene_name+"|species="+species+"|TIS_position="+str(TIS_pos)+"|annotation_source="+annotation_source+"|first_downstream_intron_start="+str(first_downstream_intron_position)+"|5_flanking_sequence_missing="+flanking_seq_missing_5+"|3_flanking_sequence_missing="+flanking_seq_missing_3+"|TSS_annotated="+TSS_annotated+"\n")
			outfile_test_sequences.write(gene_seq+"\n")

			#Print status message
			if seqs_counter % 100 == 0:
				print(f"Processed {seqs_counter}/{test_seqs_no} transcripts.")

	#Close file
	outfile_test_sequences.close()


def remove_duplicates(species, genomic_testsets_dir):
	"""
	Remove duplicate sequences with the same TIS arising from potential presence of several transcript variants of the same gene for genomic TIS test set.

	Args:
		species (str): The species name.
		genomic_testsets_dir (str): The directory to store the genomic test sets.

	Output:
		genomic_testset_{species}.fasta.gz (file): A compressed fasta file containing the genomic test set sequences without duplicates.
	"""
	
	#Initialize
	unique_headers = {}

	#Read input fasta and filter duplicates
	with gzip.open(f"../../data/data_evaluation/input_testsets/genomic_testsets/{genomic_testsets_dir}/genomic_testset_{species}.fna.gz", "rt") as input_handle:
		for record in SeqIO.parse(input_handle, "fasta"):
			#Use header to filter out duplicates
			header = record.id
			if header not in unique_headers:
				unique_headers[header] = record

	#remove original datsset
	os.remove(f"../../data/data_evaluation/input_testsets/genomic_testsets/{genomic_testsets_dir}/genomic_testset_{species}.fna.gz")

    #Write filtered sequences to file
	with gzip.open(f"../../data/data_evaluation/input_testsets/genomic_testsets/{genomic_testsets_dir}/genomic_testset_{species}.fasta.gz", "wt") as output_handle:
		for record in unique_headers.values():
            #Write header
			output_handle.write(f">{record.id}\n")
            #Write sequence in one line
			output_handle.write(f"{str(record.seq)}\n")


def extract_genomic_testset_cleaned(genomic_testsets_dir, extend_gene_end_upstream, extend_gene_end_downstream):
	#Extract the TIS-labelled sequence numbers correpsonding to test set (last partition)
	TIS_test_seqs_dict = get_test_sequences_TIS()

	#Get list of all species names 
	species_list = get_species_names()

	#Loop over each species to generate results
	for species in species_list:
		print(f"Creating testset for {species}.")
		extract_genomic_testset(species, TIS_test_seqs_dict, genomic_testsets_dir, extend_gene_end_upstream, extend_gene_end_downstream)
		print(f"Removing duplicates for {species}.")
		remove_duplicates(species, genomic_testsets_dir)


#extract_genomic_testset_cleaned("genes", 1000, 0) #1000 for estimating length of promoter region; 0 extra up- and downstream
#extract_genomic_testset_cleaned("genes_extended_1000bp", 1000+1000, 1000) #1000 for estimating length of promoter region; 1000 extra up- and downstream to represent genomic context