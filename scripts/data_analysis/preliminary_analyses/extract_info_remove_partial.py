#!/usr/bin/env python3

import gzip
import re
import json
import os
import time
import pandas as pd

class GFFProcesser:
	"""
	Process and extract useful annotations from genomic gff-files,
	store in dict with each key corresponding to information about one transcript
	"""

	def __init__(self, species):
		"""
        Initialize the GFFProcesser class.

        Args:
            species (str): The species name.
        """

		self.species = species
		self.transcripts_info_dict = {}

	def extract_correct_positions(self, transcript_id):
		"""
		Make sure that positions have been extracted correctly;
		if either exon positions or CDS positions are not annotated for mRNA transcript, then remove it

		Args:
			transcript_id (str): The particular ID of the transcript, datapoint is extracted from. 
		"""
		
		#Only relevant when a transcript ID is found; here we ignore mRNAs with partial annotations
		if transcript_id != "":
			#No annotated CDS (and thereby TIS)
			if len(self.transcripts_info_dict[transcript_id]["CDS_pos"]) == 0:
				del self.transcripts_info_dict[transcript_id]
			
			#No annotated exons (no defined mature mRNA)
			elif len(self.transcripts_info_dict[transcript_id]["exon_pos"]) == 0:
				del self.transcripts_info_dict[transcript_id]
			
			else:
				#Check that exons and mRNA are extracted correctly
				flattened_exon_positions = [item for sublist in self.transcripts_info_dict[transcript_id]["exon_pos"] for item in sublist]
				min_exon = min(flattened_exon_positions) #beginning position of first exon
				max_exon = max(flattened_exon_positions) #end position of last exon
				
				#Check that mRNA boundaries correspond to exon boundaries
				if self.transcripts_info_dict[transcript_id]["mRNA_pos"][0] != min_exon:
					del self.transcripts_info_dict[transcript_id]
				elif self.transcripts_info_dict[transcript_id]["mRNA_pos"][1] != max_exon:
					del self.transcripts_info_dict[transcript_id]


	def process_gff_file(self):
		"""
		Process GFF-file line by line to extract complete annotations on an mRNA-level
		"""
		
		#Initialize
		transcript_id = ""
		prot_name = ""
		initial_mRNA_found = False
		prot_name_found = False

		#Open gff-file with mRNA annotations
		with gzip.open(f"../../../data/data_raw/genomic_gff/{self.species}_genomic.gff.gz", "rt") as info_file:
			#Read and discard the first 8 lines (information lines)
			for _ in range(8):
					next(info_file)
			
			#Continue reading remaining of file line by line
			for line in info_file:
				#Split information from columns
				col_info = line.split("\t")

				#Skip non-annotation lines
				if len(col_info) >= 3:

					#Extract annotations from given sources
					if col_info[1] in ["BestRefSeq", "RefSeq", "Gnomon"]: 

						#Extract lines with mRNA annotation
						if col_info[2] == "mRNA":

							#Make sure that positions have been extracted correctly
							if initial_mRNA_found:
								self.extract_correct_positions(transcript_id)

							#Re-initialize for each new mRNA annotation
							prot_name = ""
							prot_name_found = False
							initial_mRNA_found = True

							#Extract attribute information column
							attr_col = col_info[8]
							match_id = re.search(r'ID=rna-([^;]+)', attr_col)				#Look for transcript ID
							if match_id:
								match_partial = re.search(r'partial=true', attr_col)		#Look for partial annotations

								#Remove partially annotated transcripts
								if match_partial:
									transcript_id = ""

								#Save information with complete annotations
								else:
									#Extract transcript id, remove potential newline-characters
									transcript_id = match_id.group(1).strip()

									#Initialize inner dict belonging to transcript id
									self.transcripts_info_dict[transcript_id] = {"mRNA_pos": [int(col_info[3]), int(col_info[4])],		#mRNA coordinates on chromosome
																				"Strand": col_info[6],									#Strand-tag
			                                                      				"exon_pos": [],											#Annotated exon start- and stop coordinates
			                                                      				"CDS_pos": [],											#Annotated CDS start- and stop coordinates
			                                                      				"chrom": col_info[0],									#Chromosome-tag
			                                                      				"source": col_info[1]}									#Algorithm used for annotated feature
							
						#Only look for annotations where parent mRNA is completely annotated
						if transcript_id != "":
							
							#Search for exon and CDS annotations
							if col_info[2] in ["exon", "CDS"]:
								
								#Extract attribute information column
								attr_col = col_info[8]

								#Look for exon- and CDS-annotations belonging to given transcript
								match_id = re.search(r'Parent=rna-([^;]+)', attr_col)
								if match_id:
									#Make sure exon only belongs to one mRNA 
									#(the reason for doing this is that I have yet only discovered such cases from the gff-files. 
									#if this eventually was not the case, the code would need to account for exon positions belonging 
									#to more than one transcript, as examplified on github https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md#parent-part_of-relationships)
									assert len(match_id.group(1).split(",")) == 1
									
									#Check whether the found Parent mRNA ID belongs to the current mRNA
									if match_id.group(1) == transcript_id:
										
										#Extract exon positions
										if col_info[2] == "exon":
											self.transcripts_info_dict[transcript_id]["exon_pos"].append([int(col_info[3]), int(col_info[4])])

										#Extract CDS positions
										elif col_info[2] == "CDS":
											self.transcripts_info_dict[transcript_id]["CDS_pos"].append([int(col_info[3]), int(col_info[4])])

											#Check if one or more proteins arise from the mRNA transcript variant; 
											#from the gff-file structures I have not ran into cases like this with the tested files, 
											#but the assertion makes sure we know if it happens and can modify to account for it 
											#(like if the same exon belongs to several mRNAs, example also from github)
											match_prot_name = re.search(r'Name=([^;]+);', attr_col)
											
											if match_prot_name:
												if not prot_name_found:
													#Store protein name from mRNA
													prot_name = match_prot_name.group(1)
												else: 
													assert prot_name == match_prot_name.group(1)
												prot_name_found = True

		#Make sure that positions have been extracted correctly for last mRNA
		self.extract_correct_positions(transcript_id)

		#Print a summary from extraction
		print("Sequence annotations harvested from species", self.species)
		print("Number of transcripts extracted:", len(self.transcripts_info_dict.keys()))

		#Save dict with annotation information
		with open(f"../../../data/data_analysis/data_preparatory_analysis/transcripts_info/{self.species}_dict.json", "w") as dict_file:
			json.dump(self.transcripts_info_dict, dict_file)



class ExtractSequenceInformation:
	"""
	Extract various information from mRNAs tat have complete annotations,
	such as 5' UTR length, start codon at TIS and source of annotation.
	This version of the script is used for initial analysis of 5' UTR lengths etc.
	to estimate likely 5' UTR lengths in species lacking annotations on 5' UTR lengths (missing transcription start site)
	"""

	def __init__(self, species):
		"""
		Initialize the ExtractSequenceInformation class.

		Args:
            species (str): The species name.
		"""

		self.species = species
		self.chromosome_dict = {}

	def load_annotation_info_dict(self):
		"""
		Load the mRNA annotation information extraced from gff-file in dict
		"""
		with open(f'../../../data/data_analysis/data_preparatory_analysis/transcripts_info/{self.species}_dict.json', 'r') as dict_file:
			annotations_dict = json.load(dict_file)

		print("Transcripts from annotation file: ", len(list(annotations_dict.keys())))

		return annotations_dict


	def reverse_complement(self, seq): 
		"""
		Take the reverse complement of a sequence for mRNAs located on the complement strand.

		Args:
			seq (str): The sequence to take reversed complement of. 
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


	def extract_strand_information(self, 
								   annotations_dict, 
								   transcript_id, 
								   chromosome_seq, 
								   TIS_information, 
								   cds_ends_correct, 
								   cds_ends_wrong):
		
		"""
		Extract sequence information on respective strand of an mRNA annotation.
		The function should be run for every individual transcript

		Args:
			annotation_dict (dict): Dict with annotations, processed from gff-file.
			transcript_id (str): The particular ID of the transcript, datapoint is extracted from. 
			chromosome_seq (str): The sequence of a landmark (scaffold, chromosome, etc. dependent on assembly).
			TIS_information (df): dataframe containing the information in the final dataset-format, for the TIS labelled ATG samples.
			cds_ends_correct (int): number of transcripts in which the coding sequence ends with an actual stop codon. 
			cds_ends_wrong (int): number of transcripts in which the coding sequence ends with another codon than a stop codon. 
		"""
		
		#Initialize for every transcript
		write_transcript_annotation = True
		exon_seq = ""
		exons_length = 0

		#Handle cases on template strand
		if annotations_dict[transcript_id]["Strand"] == "+": 

			#Sort lists of exon and CDS positions based on start position of each exon and CDS, respectively
			sorted_exons = sorted(annotations_dict[transcript_id]["exon_pos"], key=lambda x: x[0])
			sorted_CDS = sorted(annotations_dict[transcript_id]["CDS_pos"], key=lambda x: x[0])

			#Locate start position (TIS) and end position (stop codon) of CDS
			CDS_start = sorted_CDS[0][0]
			CDS_stop = sorted_CDS[-1][-1]
			
			#Loop over every exon-region in mRNA
			for exon_pos in sorted_exons:

				if exon_pos[0] <= CDS_start <= exon_pos[1]:
					#Locate TIS position in the mature mRNA (introns removed)
					TIS = exons_length + CDS_start - exon_pos[0]			#TIS: 0-index (starts in 0); corresponds to position on exons (mature mRNA)
				
				if exon_pos[0] <= CDS_stop <= exon_pos[1]:
					#Locate stop codon/end of CDS in the mature mRNA (introns removed)
					CDS_end = exons_length + CDS_stop - exon_pos[0] + 1		

				exons_length += exon_pos[1] - exon_pos[0] + 1 							#Total length of exons in mRNA
				exon_seq += chromosome_seq[exon_pos[0]-1:exon_pos[1]]					#The total exon sequence

			#Check that coding sequence corresponds to a number of triplets
			if len(exon_seq[TIS:CDS_end]) % 3 != 0: 
				write_transcript_annotation = False										#Skip annotations with errors
								
			#assure that the extracted CDS ends with stop codon (if not there is an error in annotation)
			if exon_seq[CDS_end-3:CDS_end].lower() in ["tga", "tag", "taa"]:
				cds_ends_correct += 1
			else:
				cds_ends_wrong += 1
				write_transcript_annotation = False										#Skip annotations with errors
				
			if write_transcript_annotation:
				if TIS == 0:
					utr_5 = None
				else:
					utr_5 = TIS-1
				TIS_information = pd.concat([TIS_information, pd.DataFrame({'Transcript_id': transcript_id,									#Transcript ID
																			'TIS_codon': [exon_seq[TIS:TIS+3]], 							#Codon at the TIS
																			'UTR_5_length': [utr_5], 										#Annotated 5' UTR length
																			'Source': [annotations_dict[transcript_id]["source"]]})], 		#Annotation source (BestRefseq, Gnomon, RefSeq)
				ignore_index=True)	

		#Handle cases on complement strand
		elif annotations_dict[transcript_id]["Strand"] == "-":

			#########################
			##Extract exon sequence##
			#########################

			#Sort lists of exon and CDS positions based on start position of each exon and CDS, respectively
			sorted_exons = sorted(annotations_dict[transcript_id]["exon_pos"], key=lambda x: x[0])
			sorted_CDS = sorted(annotations_dict[transcript_id]["CDS_pos"], key=lambda x: x[0])
								
			for exon_pos in sorted_exons:
				exon_seq += chromosome_seq[exon_pos[0]-1:exon_pos[1]]					#The total exon sequence seen from the template strand

			#Get complement strand sequence by taking reverse complement of sequence
			exon_seq = self.reverse_complement(exon_seq)

			#########################
			#######Extract TIS#######
			#########################

			#Sort lists of exon and CDS positions based on start position of each exon and CDS, respectively, but reversed
			sorted_exons = sorted(annotations_dict[transcript_id]["exon_pos"], key=lambda x: x[0], reverse=True)
			sorted_CDS = sorted(annotations_dict[transcript_id]["CDS_pos"], key=lambda x: x[0], reverse = True)
			
			#Locate start position (TIS) and end position (stop codon) of CDS corresponding to the ones on complement strand	
			CDS_start = sorted_CDS[0][-1]
			CDS_stop = sorted_CDS[-1][0]
								
			for exon_pos in sorted_exons:
				if exon_pos[0] <= CDS_start <= exon_pos[1]:
					#Locate TIS position in the mature mRNA (introns removed)
					TIS = exons_length + exon_pos[1] - CDS_start					#TIS; 0-index (starts in 0); corresponds to position on exons (mature mRNA)
				
				if exon_pos[0] <= CDS_stop <= exon_pos[1]:
					#Locate stop codon/end of CDS in the mature mRNA (introns removed)
					CDS_end = exons_length + exon_pos[1] - CDS_stop	+ 1				

				exons_length += exon_pos[1] - exon_pos[0] + 1 						#Total length of exons in mRNA


			#Check that coding sequence corresponds to a number of triplets
			if len(exon_seq[TIS:CDS_end]) % 3 != 0: 
				write_transcript_annotation = False									#Skip annotations with errors
				
			#Assure that the extracted CDS ends with stop codon (if not there would be an error with the code)
			if exon_seq[CDS_end-3:CDS_end].lower() in ["tga", "tag", "taa"]:
				cds_ends_correct += 1
			else:
				cds_ends_wrong += 1
				write_transcript_annotation = False									#Skip annotations with errors

			if write_transcript_annotation: 
				if TIS == 0:
					utr_5 = None
				else:
					utr_5 = TIS-1
				TIS_information = pd.concat([TIS_information, pd.DataFrame({'Transcript_id': transcript_id, 								#Transcript ID
																			'TIS_codon': [exon_seq[TIS:TIS+3]], 							#Codon at the TIS
																			'UTR_5_length': [utr_5], 										#Annotated 5' UTR length
																			'Source': [annotations_dict[transcript_id]["source"]]})], 		#Annotation source (BestRefseq, Gnomon, RefSeq)
																		    ignore_index=True)


		return TIS_information, cds_ends_correct, cds_ends_wrong
							

	def process_and_extract_sequence_info(self):
		"""
		Process and extract mRNA sequence information from an entire genomic fasta file,
		based on annotations extracted from corresponding GFF-file
		"""

		#Load dict with transcript annotations
		annotations_dict = self.load_annotation_info_dict()

		#Initialize
		chromosome_seq_initialized = False
		extract_info = True
		cds_ends_correct = 0
		cds_ends_wrong = 0

		#Create empty DataFrame with column names
		TIS_information = pd.DataFrame(columns=['Transcript_id', 
												'TIS_codon', 
												'UTR_5_length', 
												'Source'])

		#Open genome sequence in fasta format
		with gzip.open(f"../../../data/data_raw/genomic_fna/{self.species}_genomic.fna.gz", "rt") as seq_file:
			
			#Iterate line by line
			for line in seq_file:

				#Every time a new chromosome starts
				if line.startswith(">"):

					if chromosome_seq_initialized:

						#Only extract from nuclear genes
						if extract_info:

							#Initialize on new for every chromosome
							transcripts_in_chromosome = []

							#Extract all correctly annotated transcripts on currently stored chromosome
							for transcript_id, transcript_info in annotations_dict.items():
								if 'chrom' in transcript_info and transcript_info['chrom'] == chrom_tag:
									transcripts_in_chromosome.append(transcript_id)

							#Loop over every transcript seperately to extract information individually
							for i in range(len(transcripts_in_chromosome)):

								TIS_information, cds_ends_correct, cds_ends_wrong = self.extract_strand_information(annotations_dict = annotations_dict, 
																													transcript_id = transcripts_in_chromosome[i], 
																													chromosome_seq = chromosome_seq, 
																													TIS_information = TIS_information,
																													cds_ends_correct = cds_ends_correct,
																													cds_ends_wrong = cds_ends_wrong)

					entry_line = line

					#Store chromosome annotation for mapping transcripts to correct chromosome
					chrom_tag = line.split(" ")[0].strip(">")
					
					#Only store nuclear sequences
					if "mitochondrion" in line or "chloroplast" in line:
						extract_info = False
					else:
						extract_info = True

					#Re-initialize
					chromosome_seq_initialized = True
					chromosome_seq = ""
				
				#Every sequence line belonging to current chromosome is stored
				else: 
					if extract_info:
						chromosome_seq += line.strip()

			#For the last iteration (the last landmark/chromosome identifier)
			#Only extract from nuclear genes
			if extract_info:
				#Initialize on new for every chromosome
				transcripts_in_chromosome = []

				#Extract all transcripts on currently stored chromosome
				for transcript_id, transcript_info in annotations_dict.items():
					if 'chrom' in transcript_info and transcript_info['chrom'] == chrom_tag:
						transcripts_in_chromosome.append(transcript_id)

				#Loop over every transcript seperately to extract information individually
				for i in range(len(transcripts_in_chromosome)):

					TIS_information, cds_ends_correct, cds_ends_wrong = self.extract_strand_information(annotations_dict = annotations_dict, 
																										transcript_id = transcripts_in_chromosome[i],
																										chromosome_seq = chromosome_seq, 
																										TIS_information = TIS_information,
																										cds_ends_correct = cds_ends_correct,
																										cds_ends_wrong = cds_ends_wrong)
			if len(list(annotations_dict.keys())) != 0 and cds_ends_correct != 0:
				print("Wrong stop codons:", str(cds_ends_wrong), "\nCorrect stop codons:", str(cds_ends_correct))
				print("Percentage correct: ", str(cds_ends_correct/(cds_ends_correct+cds_ends_wrong) * 100))

			#Save transcript data to CSV
			TIS_information.to_csv(f'../../../data/data_analysis/data_preparatory_analysis/transcripts_data/transcript_data_{self.species}.csv', index=False)



def main():
	"""Run pipeline to execute"""

	#All the species which has less than 1 % of transcripts annotated with partial=true

	species_filenames_list = os.listdir("../../../data/data_evaluation/input_testsets/genomic_testsets")
	species_list = [filename.split(".fasta.gz")[0].split("genomic_testset_")[1] for filename in species_filenames_list]

	for species in species_list[43:]:
		print("\n\n")
		print(species)

		#Process annotations and store relevant information
		process_information = GFFProcesser(species)
		process_information.process_gff_file()

		#Extract sequence information based on processed annotations
		extractor = ExtractSequenceInformation(species)
		extractor.process_and_extract_sequence_info()


start_time = time.time()
main()
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")