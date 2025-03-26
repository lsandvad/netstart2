import gzip
import os
import pandas as pd

#Get list of all species names
species_filenames_list = os.listdir("../../data/data_raw/genomic_gff")
species_list = [filename.split("_genomic.gff.gz")[0] for filename in species_filenames_list]

#Open graphart assignments of TIS-labelled sequences
with gzip.open(f"../../data/data_model_preparation/graphpart_assignments/graphpart_assignments_TIS.csv.gz", "rt") as file:
	df_TIS_partitions = pd.read_csv(file)  

#Extract all sequences assignment to test partition for each species, store in dict with key = species, values = [seq_number_x,...,]
TIS_testset = df_TIS_partitions[df_TIS_partitions["cluster"] == 4.0] #clusters are assigned 0, 1, 2, 3, 4; 4 is test cluster
TIS_test_seqs_list = list(TIS_testset["AC"])
TIS_test_seqs_dict = {}

for seq_info in TIS_test_seqs_list:
	str_parts = seq_info.split('_')
	species_name = "_".join(str_parts[3:])
	seq_number = "_".join(str_parts[:3])

	if species_name not in TIS_test_seqs_dict:
		TIS_test_seqs_dict[species_name] = []

	#Get TIS-labelled sequences from test partition
	TIS_test_seqs_dict[species_name].append(seq_number)


###Repeat process for all non-TIS sequence types###

#Intergenic sequences
with gzip.open(f"../../data/data_model_preparation/graphpart_assignments/graphpart_assignments_intergenic.csv.gz", "rt") as file:
	df_intergenic_partitions = pd.read_csv(file)  

intergenic_testset = df_intergenic_partitions[df_intergenic_partitions["cluster"] == 4.0] #(clusters are assigned 0, 1, 2, 3, 4; 4 is test cluster)
intergenic_test_seqs_list = list(intergenic_testset["AC"])
intergenic_test_seqs_dict = {}

for seq_info in intergenic_test_seqs_list:
	str_parts = seq_info.split('_')
	species_name = "_".join(str_parts[2:])
	seq_number = "_".join(str_parts[:2])

	if species_name not in intergenic_test_seqs_dict:
		intergenic_test_seqs_dict[species_name] = []

	intergenic_test_seqs_dict[species_name].append(seq_number)


#Intron sequences
with gzip.open(f"../../data/data_model_preparation/graphpart_assignments/graphpart_assignments_intron.csv.gz", "rt") as file:
	df_intron_partitions = pd.read_csv(file)  

intron_testset = df_intron_partitions[df_intron_partitions["cluster"] == 4.0] #(clusters are assigned 0, 1, 2, 3, 4; 4 is test cluster)
intron_test_seqs_list = list(intron_testset["AC"])
intron_test_seqs_dict = {}

for seq_info in intron_test_seqs_list:
	str_parts = seq_info.split('_')
	species_name = "_".join(str_parts[3:])
	seq_number = "_".join(str_parts[:3])

	if species_name not in intron_test_seqs_dict:
		intron_test_seqs_dict[species_name] = []

	intron_test_seqs_dict[species_name].append(seq_number)


#Non-TIS mRNA sequences
mRNA_non_TIS_test_seqs_dict = {}

with gzip.open(f"../../data/data_model_preparation/graphpart_assignments/graphpart_assignments_mRNA_non_TIS_upstream.csv.gz", "rt") as file:
	df_mrna_upstream_partitions = pd.read_csv(file)  

mrna_upstream_testset = df_mrna_upstream_partitions[df_mrna_upstream_partitions["cluster"] == 4.0] #(clusters are assigned 0, 1, 2, 3, 4; 4 is test cluster)
mrna_upstream_test_seqs_list = list(mrna_upstream_testset["AC"])

for seq_info in mrna_upstream_test_seqs_list:
	str_parts = seq_info.split('_')
	species_name = "_".join(str_parts[5:])
	seq_number = "_".join(str_parts[:5])

	if species_name not in mRNA_non_TIS_test_seqs_dict:
		mRNA_non_TIS_test_seqs_dict[species_name] = []

	mRNA_non_TIS_test_seqs_dict[species_name].append(seq_number)

with gzip.open(f"../../data/data_model_preparation/graphpart_assignments/graphpart_assignments_mRNA_non_TIS_downstream.csv.gz", "rt") as file:
	df_mrna_downstream_partitions = pd.read_csv(file)  

mrna_downstream_testset = df_mrna_downstream_partitions[df_mrna_downstream_partitions["cluster"] == 4.0] #(clusters are assigned 0, 1, 2, 3, 4; 4 is test cluster)
mrna_downstream_test_seqs_list = list(mrna_downstream_testset["AC"])

for seq_info in mrna_downstream_test_seqs_list:
	str_parts = seq_info.split('_')
	species_name = "_".join(str_parts[5:])
	seq_number = "_".join(str_parts[:5])

	if species_name not in mRNA_non_TIS_test_seqs_dict:
		mRNA_non_TIS_test_seqs_dict[species_name] = []

	mRNA_non_TIS_test_seqs_dict[species_name].append(seq_number)


#Create full testset for each species
for species in species_list: 
	print(f"Creating testset in fasta format for species {species}.")

	#Open outfile
	testdata_species = gzip.open(f"../../data/data_evaluation/input_testsets/mRNA_testsets/input_testset_{species}.fasta.gz", "wt")

	#Write test sequences of all sequence types to outfile in fasta format

	#TIS-labelled mRNA sequences
	with gzip.open(f"../../data/data_model_preparation/datasets/TIS/mRNA_positive_{species}.csv.gz", "rt") as file:

		df_TIS_sequences = pd.read_csv(file)  

		TIS_test_seqs_df = df_TIS_sequences[df_TIS_sequences["Seq_number"].isin(TIS_test_seqs_dict[species])]

		for index, row in TIS_test_seqs_df.iterrows():

			fasta_header = f'>{row["species"]}|seq_number={row["Seq_number"]}|TIS={row["TIS"]}|type={row["codon_type"]}|source={row["annotation_source"]}|ATG_pos={row["ATG_position"]}|ATG_relative={row["ATG_relative"]}\n'
			fasta_sequence = f"{row['Sequence']}\n"

			testdata_species.write(fasta_header)
			testdata_species.write(fasta_sequence)

	#Non-TIS labelled mRNA sequences
	with gzip.open(f"../../data/data_model_preparation/datasets/non_TIS/mRNA/mRNA_negative_{species}.csv.gz", "rt") as file:
		df_mRNA_non_TIS_sequences = pd.read_csv(file)  

		mRNA_non_TIS_test_seqs_df = df_mRNA_non_TIS_sequences[df_mRNA_non_TIS_sequences["Seq_number"].isin(mRNA_non_TIS_test_seqs_dict[species])]

		for index, row in mRNA_non_TIS_test_seqs_df.iterrows():

			fasta_header = f'>{row["species"]}|seq_number={row["Seq_number"]}|TIS={row["TIS"]}|type={row["codon_type"]}|source={row["annotation_source"]}|ATG_pos={row["ATG_position"]}|ATG_relative={row["ATG_relative"]}\n'
			fasta_sequence = f"{row['Sequence']}\n"

			testdata_species.write(fasta_header)
			testdata_species.write(fasta_sequence)

	#Intergenic sequences
	with gzip.open(f"../../data/data_model_preparation/datasets/non_TIS/intergenic/intergenic_data_{species}.csv.gz", "rt") as file:
		df_intergenic_sequences = pd.read_csv(file)  

		intergenic_test_seqs_df = df_intergenic_sequences[df_intergenic_sequences["Seq_number"].isin(intergenic_test_seqs_dict[species])]

		for index, row in intergenic_test_seqs_df.iterrows():

			fasta_header = f'>{row["species"]}|seq_number={row["Seq_number"]}|TIS={row["TIS"]}|type={row["codon_type"]}|source={row["annotation_source"]}|ATG_pos={row["ATG_position"]}|ATG_relative={row["ATG_relative"]}\n'
			fasta_sequence = f"{row['Sequence']}\n"

			testdata_species.write(fasta_header)
			testdata_species.write(fasta_sequence)

	#Intron sequences
	with gzip.open(f"../../data/data_model_preparation/datasets/non_TIS/introns/introns_{species}.csv.gz", "rt") as file:
		df_introns_sequences = pd.read_csv(file)  

		#Some species dont contain introns; skip
		try: 
			intron_test_seqs_df = df_introns_sequences[df_introns_sequences["Seq_number"].isin(intron_test_seqs_dict[species])]

			for index, row in intron_test_seqs_df.iterrows():

				fasta_header = f'>{row["species"]}|seq_number={row["Seq_number"]}|TIS={row["TIS"]}|type={row["codon_type"]}|source={row["annotation_source"]}|ATG_pos={row["ATG_position"]}|ATG_relative={row["ATG_relative"]}\n'
				fasta_sequence = f"{row['Sequence']}\n"

				testdata_species.write(fasta_header)
				testdata_species.write(fasta_sequence)

		except KeyError:
			continue


	testdata_species.close()

