import gzip
import re
import os
import pandas as pd
import csv

def extract_additional_info():
	"""
	Extracts annotation sources and 5' UTR lengths for mRNA sequences.

	Returns:
		annotation_source_seqs_TIS (dict): TIS-labelled mRNA sequences' annotation sources.
		annotation_source_seqs_mRNA_non_TIS (dict): Non-TIS-labelled mRNA sequences' annotation sources.
		utr_5_length (dict): 5' UTR lengths for TIS-labelled mRNA sequences.
	"""

	#Initialize
	annotation_source_seqs_TIS = {}
	annotation_source_seqs_mRNA_non_TIS = {}
	utr_5_length = {}

	#Get list of all species names in dataset
	species_filenames_list = os.listdir("../../data/data_raw/genomic_gff")
	species_list = [filename.split("_genomic.gff.gz")[0] for filename in species_filenames_list]

	#Extract annotation source for each sequence in each species
	for species in species_list:
		#TIS-labelled mRNA sequences
		with gzip.open(f"../../data/data_model_preparation/datasets/TIS/mRNA_positive_{species}.csv.gz", "rt") as file:
			species_TIS_df = pd.read_csv(file)

			for index, row in species_TIS_df.iterrows():
				key = row["Seq_number"] + "_" + species
				annotation_source_seqs_TIS[key] = row["annotation_source"]
				utr_5_length[key] = row["5_UTR_length_annotated"]

		#Non-TIS-labelled mRNA sequences
		with gzip.open(f"../../data/data_model_preparation/datasets/non_TIS/mRNA/mRNA_negative_{species}.csv.gz", "rt") as file:
			species_non_TIS_df = pd.read_csv(file)

			for index, row in species_non_TIS_df.iterrows():
				key = row["Seq_number"] + "_" + species
				annotation_source_seqs_mRNA_non_TIS[key] = row["annotation_source"]
	
	return annotation_source_seqs_TIS, annotation_source_seqs_mRNA_non_TIS, utr_5_length


def extract_partitions_data(partition_writers, seq_type):
	"""
	Extract partitioned data of a sequence type and assign the correct sequences to respective partitions. 

	Args:
		partition_writers (dict): dict containing the information going into files for each partition.
		seq_type (str): the sequence type to extract partitioned datasets for. 
			Options: "TIS", "intergenic", "intron", "mRNA_non_TIS_upstream", "mRNA_non_TIS_downstream"

	Returns:
		partition_writers (dict): Updated partition_writers with sequences written to their respective partition files.
	"""
	
	print("Processing sequence type: ", seq_type, flush = True)

	partition_dict = {}
	
	#Open graphpart assignments (which sequences go into which partition)
	with gzip.open("../../data/data_model_preparation/graphpart_assignments/graphpart_assignments_"+seq_type+".csv.gz", "rt") as assignment_file:
		#Write correct sequences to each partition
		for line in assignment_file:
			partition = line.strip().split(",")[-1]
			seq_id = line.strip().split(",")[0]

			#Connect sequences to assigned partition
			partition_dict[seq_id] = partition

	counter = 0

	#Open fasta file with sequences
	with gzip.open("../../data/data_model_preparation/datasets_fasta/"+seq_type+"_sequences.fasta.gz", "rt") as seq_file:
		entry_line = None

		#Loop over each line (entry line, sequence)
		for line in seq_file:

			#Entry line
			if line.startswith(">"):
				counter += 1

				#Extract sequence ID and assigned partition number
				seq_id = line.split("|")[0].strip(">")
				partition = partition_dict.get(seq_id)

				if partition in partition_files:
					entry_line = line
				else:
					entry_line = None

				if counter % 100000 == 0:
					print(counter, flush = True)

			#Sequence line
			else:
				#Write sequences and revelant information to correct partition files
				if entry_line and partition in partition_files:
					partition_writers[partition].write(entry_line)
					partition_writers[partition].write(line)

	return partition_writers


def convert_fasta_to_csv(partition_number, annotation_source_seqs_TIS, annotation_source_seqs_mRNA_non_TIS, utr_5_length):
	"""
	Converts a gzipped FASTA file to a gzipped CSV file with specific annotations to use as model input.

	Args:
		partition_number (str): The partition number used to identify the input FASTA file and output CSV file.
		annotation_source_seqs_TIS (dict): A dictionary mapping sequence IDs to their annotation sources for TIS sequences.
		annotation_source_seqs_mRNA_non_TIS (dict): A dictionary mapping sequence IDs to their annotation sources for mRNA non-TIS sequences.
		utr_5_length (dict): A dictionary mapping sequence IDs to their 5' UTR lengths.
	"""

	#Define the gzipped CSV file name
	gzipped_csv_file_name = "../../data/data_model/datasets/data_partition_"+partition_number+".csv.gz"

	#Open the file for writing
	with gzip.open(gzipped_csv_file_name, "wt", newline="") as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(["Sequence", "TIS", "Species", "sequence_type", "seq_number", "annotation_source", "utr_5_len"])  #Write header row

		#open fasta file from respective partition
		with gzip.open("../../data/data_model_preparation/datasets_partitioned/data_partition_"+partition_number+".fasta.gz", "rt") as fasta_file:
			
			#Initialize
			sequence = ""
			TIS = None
			species_name = None
			seq_type = None
			seq_number = None
			annotation_source = None
			utr_5_len = None

			seq_type_pattern = r'type=([^|]+)'
			TIS_pattern = r'TIS=([^|]+)'


			for line in fasta_file:
				line = line.strip()

				#Entry line
				if line.startswith(">"):

					seq_type_match = re.search(seq_type_pattern, line)
					TIS_match = re.search(TIS_pattern, line)

					if seq_type_match:
						seq_type = seq_type_match.group(1)

					#Define TIS label (0 for non-TIS ATG, 1 for TIS ATG)
					if TIS_match:
						TIS = TIS_match.group(1)


					if "type=TIS" in line:
						seq_id = line.split('|')[0].split(">")[1]
						seq_number = "_".join(seq_id.split("_")[0:3])
						species_name = line.split('|')[0].split("seq")[1].split("_")[1:]
						species_name = species_name[1:]
						species_name = species_name[0].capitalize() + ' ' + ' '.join(subname.lower() for subname in species_name[1:])
						annotation_source = annotation_source_seqs_TIS[seq_id]
						utr_5_len = utr_5_length[seq_id]

					elif "mRNA_non_TIS" in line:
						seq_id = line.split('|')[0].split(">")[1]
						seq_number = "_".join(seq_id.split("_")[0:5])
						species_name = line.split('|')[0].split("seq")[1].split("_")[1:]
						species_name = species_name[1:]
						species_name = species_name[0].capitalize() + ' ' + ' '.join(subname.lower() for subname in species_name[1:])
						annotation_source = annotation_source_seqs_mRNA_non_TIS[seq_id]
						utr_5_len = ""

					elif "type=Intron" in line:
						seq_id = line.split('|')[0].split(">")[1]
						seq_number = "_".join(seq_id.split("_")[0:3])
						species_name = line.split('|')[0].split("seq")[1].split("_")[1:]
						species_name = species_name[1:]
						species_name = species_name[0].capitalize() + ' ' + ' '.join(subname.lower() for subname in species_name[1:])
						annotation_source = ""
						utr_5_len = ""

					#Intergenic sequences
					else:
						seq_id = line.split('|')[0].split(">")[1]
						seq_number = "intergenic_seq_" + seq_id.split("intergenic_seq")[1].split("_")[0]
						species_name = line.split('|')[0].split("seq")[1].split("_")[1:]
						species_name = species_name[0].capitalize() + ' ' + ' '.join(subname.lower() for subname in species_name[1:])
						annotation_source = ""
						utr_5_len = ""


				#Sequence line
				else:
					sequence += line

					assert sequence[300:303] == "ATG", "TIS ATG is not at position 301-303"

					if utr_5_len == None or utr_5_len == "":
						utr_5_len = 150

					elif utr_5_len < 150:
						replace_len = int(300 - utr_5_len)
						sequence = "N" * replace_len + sequence[replace_len:]
					
					assert len(sequence) == 603
					assert sequence[300:303] == "ATG", "TIS ATG is not at position 301-303"

					#If we have collected the sequence and information, write to CSV file
					if TIS is not None and species_name is not None and seq_type is not None and seq_number is not None:
						csv_writer.writerow([sequence, TIS, species_name, seq_type, seq_number, annotation_source, utr_5_len])

						#Re-initialize
						sequence = ""
						TIS = None
						species_name = None
						seq_type = None
						seq_number = None
						annotation_source = None
						utr_5_len = None


	print("Data has been converted and saved to", gzipped_csv_file_name, sep = " ", flush = True)


def check_sequences_duplicates(partition_number):
	"""
	Checks and removes duplicate sequences from a partitioned dataset, ensuring that sequences marked with TIS (Translation Initiation Site) are retained.

	Args:
		partition_number (str): The partition number used to identify the specific dataset file. It should be a string that corresponds to the partition file name.
	"""
	
	df_partition = pd.read_csv("../../data/data_model/datasets/data_partition_"+partition_number+".csv.gz", compression='gzip')
	print("Shape before removing duplicates across sequence types: ", df_partition.shape, flush = True)

	df_TIS = df_partition[df_partition['TIS'] == 1]
	df_partition = df_partition.drop_duplicates(subset=['Sequence'], keep=False)

	print("Shape after removing ALL duplicates across sequence types: ", df_partition.shape, flush = True)

	df_partition_correct = pd.concat([df_TIS, df_partition])
	df_partition_correct = df_partition_correct.drop_duplicates(subset=['Sequence'], keep="first")

	print("Shape after removing false duplicates across sequence types (keeping TIS): ", df_partition_correct.shape, flush = True)

	df_partition_correct.to_csv("../../data/data_model/datasets/data_partition_"+partition_number+".csv.gz", compression='gzip', index=False)


#Initialize
partition_files = {
    "0.0": "../../data/data_model_preparation/datasets_partitioned/data_partition_1.fasta.gz",
    "1.0": "../../data/data_model_preparation/datasets_partitioned/data_partition_2.fasta.gz",
    "2.0": "../../data/data_model_preparation/datasets_partitioned/data_partition_3.fasta.gz",
    "3.0": "../../data/data_model_preparation/datasets_partitioned/data_partition_4.fasta.gz",
    "4.0": "../../data/data_model_preparation/datasets_partitioned/data_partition_5.fasta.gz",
}

#Initialize dict for writing sequence information for each partition into files
partition_writers = {key: gzip.open(value, 'wt') for key, value in partition_files.items()}

#Add sequences of every sequence type to partitions
partition_writers = extract_partitions_data(partition_writers, "TIS")
partition_writers = extract_partitions_data(partition_writers, "intron")
partition_writers = extract_partitions_data(partition_writers, "intergenic") 
partition_writers = extract_partitions_data(partition_writers, "mRNA_non_TIS_upstream")
partition_writers = extract_partitions_data(partition_writers, "mRNA_non_TIS_downstream")

#Close files with sequences in partitions
for writer in partition_writers.values():
	writer.close()

annotation_source_seqs_TIS, annotation_source_seqs_mRNA_non_TIS, utr_5_length = extract_additional_info()

#Convert partitioned datasets to csv-format
print("Converting partition 1 to CSV", flush = True)
convert_fasta_to_csv("1", annotation_source_seqs_TIS, annotation_source_seqs_mRNA_non_TIS, utr_5_length)
check_sequences_duplicates("1")
print("Converting partition 2 to CSV", flush = True)
convert_fasta_to_csv("2", annotation_source_seqs_TIS, annotation_source_seqs_mRNA_non_TIS, utr_5_length)
check_sequences_duplicates("2")
print("Converting partition 3 to CSV", flush = True)
convert_fasta_to_csv("3", annotation_source_seqs_TIS, annotation_source_seqs_mRNA_non_TIS, utr_5_length)
check_sequences_duplicates("3")
print("Converting partition 4 to CSV", flush = True)
convert_fasta_to_csv("4", annotation_source_seqs_TIS, annotation_source_seqs_mRNA_non_TIS, utr_5_length)
check_sequences_duplicates("4")
print("Converting partition 5 to CSV", flush = True)
convert_fasta_to_csv("5", annotation_source_seqs_TIS, annotation_source_seqs_mRNA_non_TIS, utr_5_length)
check_sequences_duplicates("5")