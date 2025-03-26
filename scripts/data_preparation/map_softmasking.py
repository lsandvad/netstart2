import os
import gzip
import pandas as pd
import json
import re

def extract_transcript_sequence(annotations_dict,
								transcript_id, 
								chromosome_seq,
								extend_sequence):
		
	"""
	Extract sequence information on the respective strand of an mRNA annotation. 
	The function is run for every individual transcript.

	Args:
		annotation_dict (dict): Dict with annotations, processed from gff-file.
		transcript_id (str): The particular ID of the transcript, datapoint is extracted from. 
		chromosome_seq (str): The sequence of a landmark (scaffold, chromosome, etc. dependent on assembly).
		extend_sequence (boolean): Define whether or not the extracted transcript seuqence should be extended.
			(True for re-mapping non-TIS samples)
			(False for re-mapping TIS samples)

	Returns: 
		exon_seq (str): The extracted transcript.
		TIS (int): The position of the TIS in the extracted transcript.
		TSS_annotated (str): Whether or not the TSS is annotated in the gff-file.
	"""
		
	#Initialize for every transcript
	exon_seq_list = []
	exons_length = 0
	pattern = re.compile("[^ACGTagct]")
	TSS_annotated = "None"

	#Handle cases on template strand
	if annotations_dict[transcript_id]["Strand"] == "+": 

		#Sort lists of exon and CDS positions based on start position of each exon and CDS, respectively
		sorted_exons = sorted(annotations_dict[transcript_id]["exon_pos"], key=lambda x: x[0])
		sorted_CDS = sorted(annotations_dict[transcript_id]["CDS_pos"], key=lambda x: x[0])

		#Locate start position (TIS) and end position (stop codon) of CDS
		CDS_start = sorted_CDS[0][0]
		CDS_stop = sorted_CDS[-1][-1]
		exon_start = sorted_exons[0][0]
			
		#Loop over every exon-region in mRNA
		for exon_pos in sorted_exons:

			if exon_pos[0] <= CDS_start <= exon_pos[1]:
				#Locate TIS position in the mature mRNA (introns removed)
				TIS = exons_length + CDS_start - exon_pos[0]						#TIS: 0-index

			exons_length += exon_pos[1] - exon_pos[0] + 1 							#Total length of exons in mRNA				
			exon_seq_list.append(chromosome_seq[exon_pos[0]-1:exon_pos[1]])			#The total exon sequence

		#Join list of exon chunks to full exon sequence
		exon_seq = "".join(exon_seq_list)

		#Add some sequence to both ends of mRNA transcript
		if extend_sequence:
			if sorted_exons[0][0] < 500:
				exon_seq = chromosome_seq[:sorted_exons[0][0]-1] + exon_seq + chromosome_seq[sorted_exons[-1][-1]:sorted_exons[-1][-1]+500] 
			else:
				exon_seq = chromosome_seq[sorted_exons[0][0]-501:sorted_exons[0][0]-1] + exon_seq + chromosome_seq[sorted_exons[-1][-1]:sorted_exons[-1][-1]+500] 

		else:
			#Adjust extracted sequence in cases where 5' UTR length is not annotated; add 180 nucleotides upstream and note estimated 5' UTR
			if TIS == 0:
				exon_seq = chromosome_seq[(CDS_start - 1) - 180:CDS_start-1] + exon_seq
				TIS = 180
				TSS_annotated = "False"
				if CDS_start < 180:
					exon_seq = ""
			else:
				TSS_annotated = "True"


	#Handle cases on complement strand
	if annotations_dict[transcript_id]["Strand"] == "-":

		#########################
		##Extract exon sequence##
		#########################
		#Sort lists of exon and CDS positions based on start position of each exon and CDS, respectively
		sorted_exons = sorted(annotations_dict[transcript_id]["exon_pos"], key=lambda x: x[0])
		sorted_CDS = sorted(annotations_dict[transcript_id]["CDS_pos"], key=lambda x: x[0])
				
		for exon_pos in sorted_exons:
			exon_seq_list.append(chromosome_seq[exon_pos[0]-1:exon_pos[1]])					#The total exon sequence seen from the template strand

		exon_seq = ''.join(exon_seq_list)

		if not extend_sequence:
			#Extract 180 nucleotides upstream TIS (use for cases where 5' UTR length is not annotated)
			UTR_5_seq = chromosome_seq[sorted_exons[-1][-1]:sorted_exons[-1][-1] + 180]
			exons_and_5_utr_seq = exon_seq + UTR_5_seq
			#Get complement strand sequence by taking reverse complement of sequence
			exon_seq = reverse_complement(exon_seq)
			exons_and_5_utr_seq = reverse_complement(exons_and_5_utr_seq)

		#Add some sequence to both ends of mRNA transcript
		if extend_sequence:
			#Extract 180 nucleotides upstream TIS (use for cases where 5' UTR length is not annotated)
			upstream_UTR_5_seq = chromosome_seq[sorted_exons[-1][-1]:sorted_exons[-1][-1] + 500]
			downstream_UTR_3_seq = chromosome_seq[sorted_exons[0][0] - 500:sorted_exons[0][0]]

			#Get complement strand sequence by taking reverse complement of sequence
			extended_transcript_seq = reverse_complement(downstream_UTR_3_seq + exon_seq + upstream_UTR_5_seq)
			exon_seq = extended_transcript_seq
					

		#########################
		#######Extract TIS#######
		#########################
		#Sort lists of exon and CDS positions based on start position of each exon and CDS, respectively, but reversed
		sorted_exons = sorted(annotations_dict[transcript_id]["exon_pos"], key=lambda x: x[0], reverse=True)
		sorted_CDS = sorted(annotations_dict[transcript_id]["CDS_pos"], key=lambda x: x[0], reverse = True)
			
		#Locate start position (TIS) and end position (stop codon) of CDS corresponding to the ones on complement strand	
		CDS_start = sorted_CDS[0][-1]

		#Loop over every exon-region in mRNA
		for exon_pos in sorted_exons:
			if exon_pos[0] <= CDS_start <= exon_pos[1]:
				#Locate TIS position in the mature mRNA (introns removed)
				TIS = exons_length + exon_pos[1] - CDS_start					#TIS; 0-index

			exons_length += exon_pos[1] - exon_pos[0] + 1 						#Total length of exons in mRNA

		#Add some sequence to both ends of mRNA transcript
		if not extend_sequence:	
			#Adjust extracted sequence in cases where 5' UTR length is not annotated
			if TIS == 0:
				exon_seq = exons_and_5_utr_seq
				TIS = 180
				TSS_annotated = "False"
				if CDS_start < 180:
					exon_seq = ""
			else:
				TSS_annotated = "True"

			#Only extract ATG TIS data
			assert exon_seq[TIS:TIS+3].upper() == "ATG"

			#Use regular expression to find any characters other than A, C, G, or T
			result = pattern.search(exon_seq)
					
			#Only use sequences without missing nucleotide descriptions
			assert not result
				
	return exon_seq, TIS, TSS_annotated

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
					  'M': 'N',
					  'm': 'n'}
		
	#Complement sequence
	complemented_sequence = [complement_map[base] for base in seq]

	#Reverse the complemented sequence
	reversed_complement = ''.join(complemented_sequence[::-1])

	return reversed_complement

#Get list of all species names
species_filenames_list = os.listdir("../../data/data_evaluation/input_testsets/mRNA_testsets")
species_list = [filename.split(".fasta.gz")[0].split("input_testset_")[1] for filename in species_filenames_list]


#Remap uppercase test sequences to be softmasked for each species
for species in species_list:
	print(f"Remapping test set for: {species}.")

	#Get dict of transcript annotations for species
	with open(f'../../data/data_model_preparation/transcripts_info/{species}_dict.json', 'r') as dict_file:
		annotations_dict = json.load(dict_file)

	#open outfile to write softmasked test sequences to
	with gzip.open(f"../../data/data_evaluation/input_testsets/mRNA_testsets_processed/input_testset_{species}_softmasked.fasta.gz", "wt") as outfile:

		test_sequences = {}

		#Extract all test sequence numbers and their additional header information in dict
		with gzip.open(f"../../data/data_evaluation/input_testsets/mRNA_testsets/input_testset_{species}.fasta.gz", "rt") as file:
			for line in file:
				if line.startswith(">"):
					seq_number = line.split("seq_number=")[1].split("|")[0]
					test_sequences[seq_number] = line


		#Get intron testdata information
		intron_df = pd.read_csv(f"../../data/data_model_preparation/datasets/non_TIS/introns/introns_{species}.csv.gz", compression='gzip')
		intron_test_df = intron_df[intron_df["Seq_number"].isin(test_sequences.keys())]

		#Get intergenic testdata information
		intergenic_df = pd.read_csv(f"../../data/data_model_preparation/datasets/non_TIS/intergenic/intergenic_data_{species}.csv.gz", compression='gzip')
		intergenic_test_df = intergenic_df[intergenic_df["Seq_number"].isin(test_sequences.keys())]

		#Get full transcript testdata information (TIS-labelled sequences)
		TIS_df = pd.read_csv(f"../../data/data_model_preparation/datasets/TIS/mRNA_positive_{species}.csv.gz", compression='gzip')
		TIS_test_df = TIS_df[TIS_df["Seq_number"].isin(test_sequences.keys())]

		#Get transcript testdata information (non-TIS-labelled sequences)
		mRNA_non_TIS_df = pd.read_csv(f"../../data/data_model_preparation/datasets/non_TIS/mRNA/mRNA_negative_{species}.csv.gz", compression='gzip')
		mRNA_non_TIS_test_df = mRNA_non_TIS_df[mRNA_non_TIS_df["Seq_number"].isin(test_sequences.keys())]

		#Loop through genome fasta file (softmasked)
		with gzip.open(f"../../data/data_raw/genomic_fna/{species}_genomic.fna.gz", "rt") as file:

			#Initialize
			initiated = False
			
			for line in file:

				#Header lines
				if line.startswith(">"):

					#If first landmark sequence has been collected; get test sequences on current landmark
					if initiated:

						#join full landmark sequence lines
						landmark_seq = ''.join(landmark_seq_parts)

						#If TIS test sequences are present on current landmark, collect them
						if TIS_landmark_seqs_df.shape[0] != 0:
							#Loop throuh each TIS test datapoint on landmark
							for i, row in TIS_landmark_seqs_df.iterrows():

								transcript_seq_softmasked, TIS, TSS_annotated = extract_transcript_sequence(annotations_dict, row["transcript_id"], landmark_seq, extend_sequence = False)

								if len(transcript_seq_softmasked) < 30000 and transcript_seq_softmasked != "":

									#Make sure transcript sequence as been correctly extracted
									assert (transcript_seq_softmasked.upper() in row["Sequence"] or row["Sequence"] in transcript_seq_softmasked.upper()), "Neither sequence is contained within the other."
									assert transcript_seq_softmasked[TIS:TIS+3].upper() == "ATG", print(TIS, transcript_seq_softmasked, TSS_annotated)#"Transcript sequence not properly extracted!"

									#Write soft-masked version of sequence to outfile
									outfile.write(">"+row["species"]+"|seq_number="+row["Seq_number"]+"|TIS="+str(row["TIS"])+"|type="+row["codon_type"]+"|source="+row["annotation_source"]+"|ATG_pos="+str(TIS)+"|ATG_relative=0|TSS_annotated="+TSS_annotated+"\n")
									outfile.write(transcript_seq_softmasked+"\n")

								else:
									print(len(transcript_seq_softmasked))

						###Collect non-TIS-labelled test sequences from mRNA, introns and intergenic regions###
						
						#If non-TIS mRNA test sequences are present on current landmark, collect them
						if mRNA_non_TIS_landmark_seqs_df.shape[0] != 0:
							#Loop through each non-TIS mRNA test datapoint on landmark
							for i, row in mRNA_non_TIS_landmark_seqs_df.iterrows():

								#Get extended transcript sequence
								transcript_seq_softmasked_extended, TIS, TSS_annotated = extract_transcript_sequence(annotations_dict, row["transcript_id"], landmark_seq, extend_sequence = True)

								test_sequence_uppercase = row["Sequence"]

								#Get positions of test sequence in extended transcript sequence
								test_seq_start = transcript_seq_softmasked_extended.upper().find(test_sequence_uppercase)
								test_seq_end = test_seq_start + len(test_sequence_uppercase)

								#Get softmasked test sequence
								test_sequence_softmasked = transcript_seq_softmasked_extended[test_seq_start:test_seq_end]

								assert test_sequence_uppercase == test_sequence_softmasked.upper()
								assert test_sequence_softmasked[500:503].upper() == "ATG"

								#Write soft-masked version of sequence to outfile
								outfile.write(test_sequences[row["Seq_number"]].strip()+"|TSS_annotated="+TSS_annotated+"\n")
								outfile.write(test_sequence_softmasked+"\n")

						#If intergenic test sequences are present on current landmark, collect them
						if intergenic_landmark_seqs_df.shape[0] != 0:
							
							#loop over all intergenic test sequences on landmark
							for i, row in intergenic_landmark_seqs_df.iterrows():
								#Get ATG positions and full softmasked test sequence
								ATG_coords = row["ATG_genomic_position"].split("[")[1].split("]")[0].split(", ")
								softmasked_seq = landmark_seq[int(ATG_coords[0])-500:int(ATG_coords[1])+500]

								assert landmark_seq[int(ATG_coords[0]):int(ATG_coords[1])].upper() == "ATG"
								assert softmasked_seq.upper() == row["Sequence"]

								#Write soft-masked version of sequence to outfile
								outfile.write(test_sequences[row["Seq_number"]].strip()+"|TSS_annotated=None\n")
								outfile.write(softmasked_seq+"\n")

						#If intron test sequences are present on current landmark, collect them
						if intron_landmark_seqs_df.shape[0] != 0:
							
							#loop over all intron test sequences on landmark
							for i, row in intron_landmark_seqs_df.iterrows():
								#Get ATG positions
								ATG_coords = row["ATG_genomic_position"].split("[")[1].split("]")[0].split(", ")

								#template strand
								if row["strand"] == "+":
									#get full softmasked test sequence
									softmasked_seq = landmark_seq[int(ATG_coords[0])-500:int(ATG_coords[1])+500]

									assert landmark_seq[int(ATG_coords[0]):int(ATG_coords[1])].upper() == "ATG"
									assert softmasked_seq.upper() == row["Sequence"]

								#complement strand
								elif row["strand"] == "-":
									#get full softmasked test sequence
									softmasked_seq = reverse_complement(landmark_seq[int(ATG_coords[0])-500:int(ATG_coords[1])+500])
									
									assert reverse_complement(landmark_seq[int(ATG_coords[0]):int(ATG_coords[1])]).upper() == "ATG"
									assert softmasked_seq.upper() == row["Sequence"]

								#Write soft-masked version of sequence to outfile
								outfile.write(test_sequences[row["Seq_number"]].strip()+"|TSS_annotated=None\n")
								outfile.write(softmasked_seq+"\n")
						
					#Get landmark tag
					landmark_tag = line.split(" ")[0].strip(">")

					#Extract info on each sequence type
					intergenic_landmark_seqs_df = intergenic_test_df[intergenic_test_df["landmark_id"] == landmark_tag]
					intron_landmark_seqs_df = intron_test_df[intron_test_df["landmark_id"] == landmark_tag]
					TIS_landmark_seqs_df = TIS_test_df[TIS_test_df["landmark_id"] == landmark_tag]
					mRNA_non_TIS_landmark_seqs_df = mRNA_non_TIS_test_df[mRNA_non_TIS_test_df["landmark_id"] == landmark_tag]

					#Re-initialize for next landmark sequence
					initiated = True
					landmark_seq_parts = []

				else: 
					#Extract landmark sequence line
					landmark_seq_parts.append(line.strip())