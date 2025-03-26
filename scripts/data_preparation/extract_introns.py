import gzip
import re
import json
import time
import pandas as pd
import random
import multiprocessing

class ExtractIntronSequences:
	"""
	Extract intron sequences and label non-TIS ATG locations.
	"""

	def __init__(self, species, group, nts_upstream_extract, nts_downstream_extract):
		"""
        Initialize the ExtractIntronSequences class.

        Args:
            species (str): The species name (format: homo_sapiens).
            group (str): The group to which the species belongs.
            nts_upstream_extract (int): Number of nucleotides upstream to extract.
            nts_downstream_extract (int): Number of nucleotides downstream to extract.
        """

		self.species = species
		self.group = group
		self.chromosome_dict = {}
		self.nts_upstream_extract = nts_upstream_extract
		self.nts_downstream_extract = nts_downstream_extract

	def load_annotation_info_dict(self):
		"""
		Load the mRNA annotation information extraced from gff-file in dict.
		"""

		try:
			with open(f'../../data/data_model_preparation/transcripts_info/{self.species}_dict.json', 'r') as dict_file:
				annotations_dict = json.load(dict_file)

				print("Transcripts from annotation file:", len(list(annotations_dict.keys())))

				return annotations_dict
	
		except FileNotFoundError:
			print(f"Error: The annotation file for {self.species} was not found.")

			return None

		except json.JSONDecodeError as error:
			print(f"Error decoding JSON for {self.species}: {error}")

			return None

		except Exception as error:
			print(f"An unexpected error occurred for {self.species}: {error}")

			return None

	def count_positives(self):
		"""
		Count the number of positive samples (TIS located) of a species. 
		"""
		try:
			with gzip.open("../../data/data_model_preparation/datasets/TIS/mRNA_positive_"+self.species+".csv.gz", 'rb') as gzipped_file:
					
				#Iterate through the lines in the gzipped file without decompressing it, 
	    		#count number of positive samples
				positive_samples = sum(1 for line in gzipped_file) - 1 #-1 due to information row

			return positive_samples

		except FileNotFoundError:
			print(f"Error: The annotation file for {self.species} was not found.")

			return None

		except Exception as error:
			print(f"An unexpected error occurred for {self.species}: {error}")

			return None


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
								   intron_information):

		"""
		Extract sequence information on the respective strand of an intron.
		The function is run for every intron observed.

		Args:
			annotation_dict (dict): Dict with annotations, processed from gff-file.
			transcript_id (str): The particular ID of the transcript, datapoint is extracted from. 
			chromosome_seq (str): The sequence of a landmark (scaffold, chromosome, etc. dependent on assembly).
			intron_information (df): dataframe containing the information in the final dataset-format, for the non-TIS ATG labelled intron samples.
		"""
		
		try:
			#Initialize for every transcript
			count_coords = 0
			introns_coords = []
			pattern = re.compile("[^ACGTagct]")

			#Handle cases on template strand
			if annotations_dict[transcript_id]["Strand"] == "+": 

				#Sort lists of exon and CDS positions based on start position of each exon and CDS, respectively
				sorted_exons = sorted(annotations_dict[transcript_id]["exon_pos"], key=lambda x: x[0])

				#Collect intron coordinates
				for i in range(len(sorted_exons) - 1):
					introns_coords.append([sorted_exons[i][1]-1,sorted_exons[i+1][0]])
				
				#Shuffle every pair of intron coordinates (randomize which intron is "looked at" first, 
				#as we only need subsample and want it to be randomized)
				random.shuffle(introns_coords)

				for coords in introns_coords:
					#Ensure that that intron coordinates are not placed to close to the end of a landmark.
					#We want to be able to extract 500 nucleotides upstream and downstream an ATG, respectively.
					if coords[1] + self.nts_downstream_extract < len(chromosome_seq):
						if coords[0] - self.nts_upstream_extract > 0:
							#Extract intron sequence
							intron_seq = chromosome_seq[coords[0]:coords[1]]
								
							#Find positions with "ATG" (0-indexed); adjust a "frame" of 500 nucleotides in both ends, 
							#in cases where introns are relatively short and we need some of the surrounding sequence
							atg_indices = [i for i in range(len(intron_seq)) if intron_seq.upper().startswith("ATG", i)]

							random.shuffle(atg_indices)

							#Extract "framed" intron sequence
							intron_seq_framed = chromosome_seq[coords[0]-self.nts_upstream_extract:coords[1]+self.nts_downstream_extract]

							#Loop over every ATG index found in randomized order
							for atg in atg_indices:

								atg_adjusted = atg + self.nts_upstream_extract
								
								#Make sure that the extracted sequence corresponds to number of upstream nts, ATG, and downstream nts
								assert len(intron_seq_framed[atg_adjusted-self.nts_upstream_extract:atg_adjusted+3+self.nts_downstream_extract]) \
									   == self.nts_upstream_extract + 3 + self.nts_downstream_extract, \
									   "Intron sequence not properly extracted."
								
								#Extract intron subsequence	
								intron_subseq = intron_seq_framed[atg_adjusted-self.nts_upstream_extract:atg_adjusted+3+self.nts_downstream_extract]

								#Make sure that the extracted sequence corresponds to number of upstream nts, ATG, and downstream nts
								assert intron_subseq[self.nts_upstream_extract:self.nts_upstream_extract+3].upper() == "ATG", \
									   "ATG-label not extracted properly."

								#Use a regular expression to find any characters other than A, C, G, or T
								result = pattern.search(intron_subseq)

								#Only include samples without missing nucleotide information
								if not result:
									#Store ATG position in sequence
									atg_pos = self.nts_upstream_extract #0-indexing in dataframe

									genomic_ATG_coords = [coords[0]+atg,coords[0]+atg+3]

									#Make sure that genomic ATG-labelled position corresponds to ATG
									assert chromosome_seq[genomic_ATG_coords[0]:genomic_ATG_coords[1]].upper() == "ATG", \
										"ATG not placed correctly in genome."
									
									#Save all relevant information in dataframe
									intron_information = pd.concat([intron_information, 
													  pd.DataFrame({'Sequence': [intron_subseq.upper()],					 						#The extracted non-coding sequence
																	'ATG_position': atg_pos,		 												#Position of ATG to predict on
																	'TIS': 0,																		#FALSE
																	'codon_type': "Intron",															#"Intron"
																	'stop_codon': None,																#None
																	'species': self.species,														#species
																	'group': self.group,															#eukaryotic grouping
																	'annotation_source': None,														#None
																	'5_UTR_length_annotated': None,													#None
																	'ATG_relative': None,															#None
																	'landmark_id': [annotations_dict[transcript_id]["chrom"]],						#Chromosome/landmark ID
																	'mrna_pos': [annotations_dict[transcript_id]["mRNA_pos"]],						#Positions of mRNA 
																	'transcript_id': transcript_id,													#Transcript ID
																	'strand': [annotations_dict[transcript_id]["Strand"]],						 	#Strand
																	'ATG_genomic_position':	[[genomic_ATG_coords[0],genomic_ATG_coords[1]]],		#Genomic position of ATG
																	'gene':	[annotations_dict[transcript_id]["gene"]],								#the name of the gene that intron arises from
																	'gene_coordinates': [annotations_dict[transcript_id]["gene_coordinates"]]		#gene coordinates
																	})],				
																	ignore_index=True)
									
									#Store only one sample from a given intron sequence per mRNA (positive label)
									break
							
							count_coords += 1

							if count_coords > 1:
								break
					

			#Handle cases on complement strand
			elif annotations_dict[transcript_id]["Strand"] == "-":

				#Sort lists of exon and CDS positions based on start position of each exon and CDS, respectively
				sorted_exons = sorted(annotations_dict[transcript_id]["exon_pos"], key=lambda x: x[0])

				for i in range(len(sorted_exons) - 1):
					introns_coords.append([sorted_exons[i][1]-1,sorted_exons[i+1][0]])
					
				random.shuffle(introns_coords)

				for coords in introns_coords:
					#Ensure that that intron coordinates are not placed to close to the end of a landmark.
					#We want to be able to extract 500 nucleotides upstream and downstream an ATG, respectively.
					if coords[1] + self.nts_downstream_extract < len(chromosome_seq):
						if coords[0] - self.nts_upstream_extract > 0:
							
							#Extract intron sequence and framed intron sequence 
							intron_seq = self.reverse_complement(chromosome_seq[coords[0]:coords[1]])
							intron_seq_framed = self.reverse_complement(chromosome_seq[coords[0]-self.nts_upstream_extract:coords[1]+self.nts_upstream_extract])

							#Find positions with "ATG" (0-indexed); adjust a "frame" of 500 nucleotides in both ends, 
							#in cases where introns are relatively short and we need some of the surrounding sequence
							atg_indices = [i for i in range(len(intron_seq)) if intron_seq.upper().startswith("ATG", i)]
							random.shuffle(atg_indices)

							#Loop over every ATG index found
							for atg in atg_indices:

								atg_adjusted = atg + self.nts_upstream_extract

								#Make sure that the extracted sequence corresponds to number of upstream nts, ATG, and downstream nts
								assert len(intron_seq_framed[atg_adjusted-self.nts_upstream_extract:atg_adjusted+3+self.nts_downstream_extract]) \
									   == self.nts_upstream_extract + 3 + self.nts_downstream_extract, \
									   "Intron sequence not properly extracted."
								
								#Extract intron subsequence	
								intron_subseq = intron_seq_framed[atg_adjusted-self.nts_upstream_extract:atg_adjusted+3+self.nts_downstream_extract]

								#Make sure that the extracted sequence corresponds to number of upstream nts, ATG, and downstream nts
								assert intron_subseq[self.nts_upstream_extract:self.nts_upstream_extract+3].upper() == "ATG", \
									   "ATG-label not extracted properly."
								
								#Use a regular expression to find any characters other than A, C, G, or T
								result = pattern.search(intron_subseq)

								#Only include samples without missing nucleotide information
								if not result:

									genomic_ATG_coords = [coords[1]-atg-3,coords[1]-atg]

									#make sure that genomic ATG-labelled position corresponds to complement reversed codon CAT
									assert chromosome_seq[genomic_ATG_coords[0]:genomic_ATG_coords[1]].upper() == "CAT", \
										"ATG not placed correctly in genome."

									#Store ATG position in sequence
									atg_pos = self.nts_upstream_extract #0-indexing in dataframe

									#Save all relevant information in dataframe
									intron_information = pd.concat([intron_information, 
																	pd.DataFrame({'Sequence': [intron_subseq.upper()], 												#The extracted non-coding sequence
																					'ATG_position': atg_pos,		 												#Position of ATG to predict on
																					'TIS': 0,																		#FALSE
																					'codon_type': "Intron",															#"Intron"
																					'stop_codon': None,																#None
																					'species': self.species,														#species
																					'group': self.group,															#eukaryotic grouping
																					'annotation_source': None,														#None
																					'5_UTR_length_annotated': None,													#None
																					'ATG_relative': None,															#None
																					'landmark_id': [annotations_dict[transcript_id]["chrom"]],						#Chromosome/landmark ID
																					'mrna_pos': [annotations_dict[transcript_id]["mRNA_pos"]],						#Positions of mRNA 
																					'transcript_id': transcript_id,													#Transcript ID
																					'strand': [annotations_dict[transcript_id]["Strand"]],						 	#Strand
																					'ATG_genomic_position':	[[genomic_ATG_coords[0],genomic_ATG_coords[1]]],		#Genomic position of ATG
																					'gene':	[annotations_dict[transcript_id]["gene"]],								#the name of the gene that intron arises from
																					'gene_coordinates': [annotations_dict[transcript_id]["gene_coordinates"]]		#gene coordinates
																					})],
																					ignore_index=True)
									
								#Store only one sample from a given intron sequence per mRNA (positive label)
								break
							
							count_coords += 1

							if count_coords > 2:
								break
						

			return intron_information

		except AssertionError as assertion_error:
			if len(intron_seq_framed[atg-self.nts_upstream_extract:atg+3+self.nts_downstream_extract]) != 0:
				print(f"Assertion error: {assertion_error}")

			return intron_information

		except Exception as error:
			print(f"An unexpected error occurred for {self.species}: {error}")
		

	def process_and_extract_sequences(self):
		"""
		Process and extract intron sequence information from an entire genomic fasta file,
		based on annotations extracted from corresponding gff-file
		"""
		try:
			#Load dict with transcript annotations
			annotations_dict = self.load_annotation_info_dict()

			#Load count of positive samples
			positive_samples = self.count_positives()

			#Initialize
			chromosome_seq_initialized = False
			extracted_all = False
			extract_info = True
			counter = 0
			transcripts_no = len(annotations_dict.keys())

			#Create empty DataFrame with column names
			intron_information = pd.DataFrame(columns=['Sequence', 				#The extracted non-coding sequence
													'ATG_position',		 		#Position of ATG to predict on
													'TIS',						#False (0)
													'codon_type',				#"Intron"
													'stop_codon',				#None
													'species',					#species
													'group',					#eukaryotic grouping
													'annotation_source',		#None
													'5_UTR_length_annotated',	#None
													'ATG_relative',				#None
													'landmark_id',				#Chromosome/landmark ID
													'mrna_pos',					#None
													'transcript_id',			#None
													'strand',					#None
													'ATG_genomic_position',		#Genomic position of ATG
													'gene',						#the name of the gene that intron arises from
													'gene_coordinates'			#gene coordinates
													])					
			
			#Open genome sequence in fasta format
			with gzip.open(f"../../data/data_raw/genomic_fna/{self.species}_genomic.fna.gz", "rt") as seq_file:
				
				#Iterate line by line
				for line in seq_file:

					#Every time a new chromosome starts
					if line.startswith(">"):
						if counter > positive_samples:
							extracted_all = True
							break

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

									intron_information = self.extract_strand_information(annotations_dict = annotations_dict, 
																						 transcript_id = transcripts_in_chromosome[i], 
																						 chromosome_seq = chromosome_seq, 
																						 intron_information = intron_information)

									if counter % 500 == 0:
										print("Processed ", counter, "/", transcripts_no, "transcripts in species ", self.species, flush = True)

									counter += 1

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
				if extract_info and not extracted_all:
					#Initialize on new for every chromosome
					transcripts_in_chromosome = []

					#Extract all transcripts on currently stored chromosome
					for transcript_id, transcript_info in annotations_dict.items():
						if 'chrom' in transcript_info and transcript_info['chrom'] == chrom_tag:
							transcripts_in_chromosome.append(transcript_id)

					#Loop over every transcript seperately to extract information individually
					for i in range(len(transcripts_in_chromosome)):

						intron_information = self.extract_strand_information(annotations_dict = annotations_dict, 
																			 transcript_id = transcripts_in_chromosome[i],
																			 chromosome_seq = chromosome_seq, 
																			 intron_information = intron_information)
						if counter % 500 == 0:
							print("Processed ", counter, "/", transcripts_no, "transcripts in species ", self.species, flush = True)
						counter += 1


			#Adjust sample size to correspond to number of positive samples
			if len(intron_information) > positive_samples:
				intron_information = intron_information.sample(n=positive_samples, random_state=42)

			#For TIS_information DataFrame
			intron_information['Seq_number'] = ['intron_seq_{}'.format(i) for i in range(1, len(intron_information) + 1)]

			#Save ATG-labelled intron dataset to CSV
			intron_information.to_csv(f'../../data/data_model_preparation/datasets/non_TIS/introns/introns_{self.species}.csv.gz', index=False, compression = 'gzip')

		except FileNotFoundError:
			print(f"Error: The genomic fasta file for {self.species} was not found.", flush = True)

		except PermissionError:
			print(f"Error: Permission denied to access the {self.species} genomic fasta file.", flush = True)

		except Exception as error:
			print(f"An unexpected error occurred while processing the species {self.species} genomic fasta file: {error}", flush = True)


def process_species(species, group):
	print("\n\n")
	print(species)
	extractor = ExtractIntronSequences(species,
									   group,
									   nts_upstream_extract = 500,
									   nts_downstream_extract = 500)
	extractor.process_and_extract_sequences()

def main():
	"""
	Run pipeline for several species
	"""

	#Extract dict with species and belonging group
	#Read the CSV file into a DataFrame
	df_species = pd.read_csv("../../data/data_raw/species_information/species_groups.csv")

	#Create a dictionary from the DataFrame with "Species" as keys and "Group" as values
	species_dict = df_species.set_index('Species')['Group'].to_dict()

	#Create a multiprocessing Pool with the number of processes you want to run concurrently
	num_processes = multiprocessing.cpu_count()  # You can adjust this as needed
	pool = multiprocessing.Pool(processes=num_processes)

	#Use multiprocessing to process species concurrently
	for species, group in species_dict.items():
		pool.apply_async(process_species, args=(species, group))

	# Close the pool and wait for all processes to finish
	pool.close()
	pool.join()

start_time = time.time()
main()
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")