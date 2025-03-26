import pandas as pd
import subprocess


def extract_subsequence(row):
	"""
	Extract subsequence in a given window surrounding labelled ATG.
	"""
	sequence = row['Sequence']
	atg_position = row['ATG_position']
    
    #Define the window size for the subsequence; 300 nucleotides upstream and downstream the ATG
    #Extract the subsequence centered around the ATG position
	subsequence = sequence[max(0, atg_position - 300): atg_position + 3 + 300]
	
	return subsequence


def clean_and_partition_sequences(sequence_type):
	"""
	Extract sequence of a predefined number of nucleotides upstream and downstream an ATG, save in fastafile. 
	Fastafile is used as input for GraphPart. 
	The function carries out Homology Partitioning on each sequence type in dataset.

	Args:
		sequence_type (str): The sequence type of the dataset to carry out homology partitioning on
			Options: "TIS", "intergenic", "intron", "mRNA_non_TIS_upstream", "mRNA_non_TIS_downstream" 
	"""

	#Make sure that input is given correctly
	if sequence_type not in ["TIS", "intergenic", "intron", "mRNA_non_TIS_upstream", "mRNA_non_TIS_downstream"]:
		raise ValueError("Invalid sequence_type. Must be one of: 'TIS', 'intergenic', 'intron', 'mRNA_non_TIS_upstream', 'mRNA_non_TIS_downstream'.")

	#Load species information and store
	df_species_details = pd.read_csv("../../data/data_raw/species_information/species_groups.csv")
	species_dict = df_species_details.set_index('Species')["Model_groups"].to_dict()
	species_list = list(species_dict.keys())

	#Initialize
	df_all_species = pd.DataFrame()
	split_data = False

	#Mark subpaths to locate correct data
	if sequence_type == "TIS":
		subpath = "TIS/mRNA_positive_"
		TIS = 1
	elif sequence_type == "intergenic":
		subpath = "non_TIS/intergenic/intergenic_data_"
		TIS = 0
	elif sequence_type == "intron":
		subpath = "non_TIS/introns/introns_"
		TIS = 0
	elif sequence_type in ["mRNA_non_TIS_upstream", "mRNA_non_TIS_downstream"]:
		subpath = "non_TIS/mRNA/mRNA_negative_"
		TIS = 0
		split_data = True
		if sequence_type == "mRNA_non_TIS_upstream":
			sequence_position = "Upstream"
		else: 
			sequence_position = "Downstream"

	#Merge sequence data for all species
	for species in species_list:
		print(species, flush = True)
		df_species = pd.read_csv("../../data/data_model_preparation/datasets/"+subpath+species+".csv.gz", compression='gzip')
		df_species['Model_groups'] = species_dict[species]

		if df_species.shape[0] > 0:

			#Apply the function to extract the subsequence
			df_species['Subsequence'] = df_species.apply(extract_subsequence, axis=1)

			#Remove subsequence column
			df_species = df_species.drop_duplicates(subset=['Subsequence'])
			df_species = df_species.drop('Subsequence', axis=1)

			df_all_species = pd.concat([df_all_species, df_species])


	print("Merged all species data. Shape:", df_all_species.shape, flush = True)

	i = 0

	#If-statement for upstream and downstream mRNA sequences
	if split_data == True:
		#Generate fastafile with all sequences
		with open('../../data/data_model_preparation/datasets_fasta/'+sequence_type+'_sequences.fasta', 'w') as outfile:
			for row in df_all_species.itertuples():
				if row.codon_type.startswith(sequence_position):
					assert row.Sequence[row.ATG_position:row.ATG_position+3] == "ATG"

					seq = row.Sequence[row.ATG_position-300:row.ATG_position+3+300]

					if len(seq) == 300 + 3 + 300:
						i += 1

						#label corresponds to organism group
						outfile.write(">"+row.Seq_number+"_"+row.species+"|label="+str(row.Model_groups)+"|TIS="+str(TIS)+"|type="+row.codon_type+"\n")
						outfile.write(seq+"\n")

	else:
		#Generate fastafile with all sequences
		with open('../../data/data_model_preparation/datasets_fasta/'+sequence_type+'_sequences.fasta', 'w') as outfile:
			for row in df_all_species.itertuples():
				assert row.Sequence[row.ATG_position:row.ATG_position+3] == "ATG"

				seq = row.Sequence[row.ATG_position-300:row.ATG_position+3+300]

				if len(seq) == 300 + 3 + 300:
					i += 1

					#label corresponds to organism group
					outfile.write(">"+row.Seq_number+"_"+row.species+"|label="+str(row.Model_groups)+"|TIS="+str(TIS)+"|type="+row.codon_type+"\n")
					outfile.write(seq+"\n")

	print("Finished creating fastafile.", flush = True)

	#Run graphpart
	graphpart_command = f"graphpart mmseqs2 -nu --fasta-file ../../data/data_model_preparation/datasets_fasta/"+sequence_type+"_sequences.fasta --threshold 0.5 --out-file ../../data/data_model_preparation/graphpart_assignments/graphpart_assignments_"+sequence_type+".csv --labels-name label --partitions 5"
	
	#Execute command
	subprocess.call(graphpart_command, shell=True)

	#Gzip fastafile
	gzip_command = f"gzip ../../data/data_model_preparation/datasets_fasta/"+sequence_type+"_sequences.fasta"
	subprocess.call(gzip_command, shell=True)


#Run graphpart on TIS mRNA sequences
clean_and_partition_sequences(sequence_type = "TIS")

#Run graphpart on sequences in intergenic regions
clean_and_partition_sequences(sequence_type = "intergenic")

#Run graphpart on non-TIS mRNA sequences upstream TIS
clean_and_partition_sequences(sequence_type = "mRNA_non_TIS_upstream")

#Run graphpart on non-TIS mRNA sequences downstream TIS
clean_and_partition_sequences(sequence_type = "mRNA_non_TIS_downstream")

#Run graphpart on intron sequences
clean_and_partition_sequences(sequence_type = "intron")
