import gzip
import re
import json
import time
import pandas as pd
import multiprocessing
import random

class ExtractIntergenicSequences:
	"""
	Extracts intergenic sequences and labels non-TIS ATGs.
	"""
	
	def __init__(self, species, group, nts_upstream_extract, nts_downstream_extract):
		"""
        Initialize the ExtractIntergenicSequences class.

        Args:
            species (str): The species name (format: homo_sapiens).
            group (str): The group to which the species belongs.
            nts_upstream_extract (int): Number of nucleotides upstream to extract.
            nts_downstream_extract (int): Number of nucleotides downstream to extract.
        """

		self.species = species
		self.group = group
		self.transcripts_info_dict = {}
		self.nts_upstream_extract = nts_upstream_extract
		self.nts_downstream_extract = nts_downstream_extract

	def count_positives(self):
		"""
		Count the number of positive samples (TIS-labelled) of a species. 
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


	def get_intragenic_coords(self):
		"""
		Get a list of all intergenic coordinates from the template strand. 

		Returns:
			intragenic_coords_landmark (dict): A dictionary with landmark IDs as keys and lists of gene coordinates as values.
		"""

		#Initialize
		intragenic_coords_landmark = {}

		try:
			with open(f'../../data/data_model_preparation/transcripts_info/'+self.species+'_dict.json', 'r') as dict_file:
				annotations_dict = json.load(dict_file)

				#Collect genetic coordinates of genes, strand and landmark annotation comes from
				for key, value in annotations_dict.items():
					gene_coords = value.get('gene_coordinates')
					strand = value.get('Strand')
					landmark = value.get('chrom')

					#Take samples from template strand
					if strand == "+":

						#Initialize for landmark
						if landmark not in intragenic_coords_landmark.keys():
							intragenic_coords_landmark[landmark] = []
							intragenic_coords_landmark[landmark].append(gene_coords)		#append gene coordinates
						else:
							intragenic_coords_landmark[landmark].append(gene_coords)		#append gene coordinates

			return intragenic_coords_landmark

		except FileNotFoundError:
			print(f"Error: The genomic fasta file for {self.species} was not found.")

			return None

		except PermissionError:
			print(f"Error: Permission denied to access the genomic fasta file for {self.species}.")

			return None

		except Exception as error:
			print(f"An unexpected error occurred while processing the genomic fasta file for {self.species}: {error}")

			return None

	def merge_intervals(self, gene_coords):
		"""
		Merge overlapping or extended gene coordinates to ensure extracting correct intergenic intervals. 

		Args:
			gene_coords (list of lists): A list of gene coordinates, where each coordinate is a list containing start and end positions [start, end].

		Returns:
			intragenic_regions (list of lists): A list of merged gene coordinates, sorted by the start position.
		"""

		#Sort the intervals based on the lowest gene start coordinate
		gene_coords.sort() 
		#Initialize with first gene's coordinates
		merged = [gene_coords[0]]

		for current in gene_coords[1:]:
			last = merged[-1]
			#Check if there's overlap
			if current[0] <= last[1]:  
				 #Merge intervals if overlapping
				last[1] = max(last[1], current[1])
			else:
				#No overlap, add the current interval
				merged.append(current) 

		#Doubble-check; sort by the first element of each sublist
		intragenic_regions = sorted(merged, key=lambda x: x[0])

		return intragenic_regions


	def collect_intergenic_samples(self, 
								intergenic_information,
								intragenic_regions_dict,
								chromosome_seq,
								landmark_tag,
								counter_samples,
								count_positives):
		"""
		Extract non-coding sequences of a certain length with an ATG and store information used for dataset.

		Args:
			intergenic_information (df): dataframe containing the information in the final dataset-format, for the non-TIS ATG labelled intergenic samples.
			intragenic_regions_dict (dict): dictionary containing information about intragenic region coordinates on a specific landmark (key: landmark, value: gene start:stop coordinates).
			chromosome_seq (str): The sequence of a landmark (scaffold, chromosome, etc. dependent on assembly).
			landmark_tag (str): The specific landmark ID
			counter_samples (int): count how many sequences have been added to dataset.
			count_positives (int): the number of TIS-labelled sequences from the given species. 

		Returns:
			intergenic_information (df): Updated dataframe with the extracted intergenic samples.
			counter_samples (int): Updated count of how many sequences have been added to the dataset.
		"""

		try:

			intragenic_regions = self.merge_intervals(intragenic_regions_dict[landmark_tag])

			genes_landmark = len(intragenic_regions)
			
			if genes_landmark > 1:
				#Loop over coordinates from every intragenic region
				for i in range(genes_landmark - 1):

					#Initialize
					counter_samples_in_region = 0
					pattern = re.compile("[^ACGTagct]")
					
					#Extract coordinates of intergenic region
					intergenic_region_start = int(intragenic_regions[i][1])
					intergenic_region_stop = int(intragenic_regions[i+1][0])

					#Extract sequence of intergenic region (-1 gives 0-index)
					intergenic_seq = chromosome_seq[intergenic_region_start-1:intergenic_region_stop-1].upper()

					#Find positions with "ATG" (0-indexed)
					atg_indices = [i for i in range(len(intergenic_seq)) if intergenic_seq.startswith("ATG", i)]
					
					random.shuffle(atg_indices)

					#Loop over every ATG index found
					for atg in atg_indices:
							
						#Only use non-coding sequences with X nts upstream and Y nts downstream an ATG
						if atg > self.nts_upstream_extract and len(intergenic_seq) - (atg + 3) > self.nts_downstream_extract:

							#Make sure that the extracted sequence corresponds to number of upstream nts, ATG, and downstream nts
							assert len(intergenic_seq[atg-self.nts_upstream_extract:atg+3+self.nts_downstream_extract]) \
								   == self.nts_upstream_extract + 3 + self.nts_downstream_extract, \
								   "Intergenic sequence not properly extracted"
								
							#Extract the non-coding subsequence we want for dataset		
							intergenic_sample = intergenic_seq[atg-self.nts_upstream_extract:atg+3+self.nts_downstream_extract]

							#Make sure that the extracted sequence corresponds to number of upstream nts, ATG, and downstream nts
							assert intergenic_sample[self.nts_upstream_extract:self.nts_upstream_extract+3].upper() == "ATG", \
								   "ATG-label not extracted properly."

							assert chromosome_seq[intergenic_region_start-1+atg:intergenic_region_start-1+atg+3].upper() == "ATG", \
									"Genomic coordinates not extractd properly."

							genomic_ATG_coords = [intergenic_region_start-1+atg,intergenic_region_start-1+atg+3]

							#Use a regular expression to find any characters other than A, C, G, or T
							result = pattern.search(intergenic_sample)

							if not result:
								#Store ATG position in sequence
								atg_pos = self.nts_upstream_extract #0-indexing in dataframe

								#Save all relevant information in dataframe
								intergenic_information = pd.concat([intergenic_information, 
					 								  pd.DataFrame({'Sequence': [intergenic_sample.upper()],									 	#The extracted non-coding sequence
																	'ATG_position': atg_pos,												 		#Position of ATG to predict on
																	'TIS': 0,																		#False (0)
																	'codon_type': "Intergenic",														#"Intergenic"
																	'stop_codon': None,																#None
																	'species': self.species,														#species
																	'group': self.group,															#eukaryotic grouping; for now, None
																	'annotation_source': None,														#None
																	'5_UTR_length_annotated': None,													#None
																	'ATG_relative': None,															#None
																	'landmark_id': landmark_tag,													#Chromosome/landmark ID
																	'mrna_pos': None,																#None
																	'transcript_id': None,															#None
																	'ATG_genomic_position':	[[genomic_ATG_coords[0],genomic_ATG_coords[1]]],		#Genomic position of ATG
																	'gene':	None,
																	'gene_coordinates': None
																	})],				
																	ignore_index=True)

								counter_samples += 1
								counter_samples_in_region += 1

								if counter_samples % 100 == 0:
									print("Processed ", counter_samples, "/", count_positives, " samples in species ", self.species, flush = True)
									
								if counter_samples_in_region > 2:
									break #only store 2 sequence per intergenic region fulfilling criteria of certain length
								

			return intergenic_information, counter_samples

		except AssertionError as assertion_error:
			print(f"Assertion error for {self.species}: {assertion_error}")

		except Exception as error:
			print(f"An unexpected error occurred for {self.species}: {error}")


	def extract_intergenic_samples(self):
		"""
		Extract non-coding sequences from genomic fasta file.
		Save sequence information to file.
		"""

		#Load dict with intragenic coordinates for each landmark
		intragenic_regions_dict = self.get_intragenic_coords()

		#Load count of positive samples
		positive_samples = self.count_positives()

		#Initialize
		chromosome_seq_initialized = False
		extract_info = True
		extracted_all = False
		chromosome_seq = ""
		counter_samples = 0
		intergenic_information = pd.DataFrame(columns=['Sequence', 					#The extracted non-coding sequence
													   'ATG_position',		 		#Position of ATG to predict on
													   'TIS',						#False (0)
													   'codon_type',				#"Intron"
													   'stop_codon',				#None
													   'species',					#species
													   'group',						#eukaryotic grouping
													   'annotation_source',			#None
													   '5_UTR_length_annotated',	#None
													   'ATG_relative',				#None
													   'landmark_id',				#Chromosome/landmark ID
													   'mrna_pos',					#None
													   'transcript_id',				#None
													   'strand',					#None
													   'ATG_genomic_position',		#Genomic position of ATG
													   'gene',						#the name of the gene that intron arises from
													   'gene_coordinates'			#gene coordinates
														])					

		try:
			#Open genome sequence in fasta format
			with gzip.open(f"../../data/data_raw/genomic_fna/{self.species}_genomic.fna.gz", "rt") as seq_file:
				
				#Iterate line by line
				for line in seq_file:
					
					#Every time a new landmark starts
					if line.startswith(">"):
						if counter_samples > positive_samples:
							extracted_all = True
							break

						if chromosome_seq_initialized:

							#Only extract from nuclear genes
							if extract_info and landmark_tag in intragenic_regions_dict.keys():
								#add non-coding sequences from one chromosome
								intergenic_information, counter_samples = self.collect_intergenic_samples(intergenic_information = intergenic_information,
																				intragenic_regions_dict = intragenic_regions_dict,
									 											chromosome_seq = chromosome_seq,
									 											landmark_tag = landmark_tag,
									 											counter_samples = counter_samples,
									 											count_positives = positive_samples)

						#Store chromosome annotation for mapping transcripts to correct chromosome
						landmark_tag = line.split(" ")[0].strip(">")
						
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

			#Repeat for last landmark (chromosome/scaffold etc.)
			if extract_info and landmark_tag in intragenic_regions_dict.keys() and not extracted_all:
				intergenic_information, counter_samples = self.collect_intergenic_samples(intergenic_information = intergenic_information,
																		intragenic_regions_dict = intragenic_regions_dict,
											 							chromosome_seq = chromosome_seq,
											 							landmark_tag = landmark_tag,
									 									counter_samples = counter_samples,
									 									count_positives = positive_samples)


			print("positive samples: ", positive_samples)
			print("intergenic samples before: ", len(intergenic_information))

			#Adjust sample size to correspond to number of positive samples
			if len(intergenic_information) > positive_samples:
				intergenic_information = intergenic_information.sample(n=positive_samples, random_state=42)

			print("intergenic samples after: ", len(intergenic_information))

			#For TIS_information DataFrame
			intergenic_information['Seq_number'] = ['intergenic_seq{}'.format(i) for i in range(1, len(intergenic_information) + 1)]

			#Save non-coding data to CSV
			intergenic_information.to_csv(f'../../data/data_model_preparation/datasets/non_TIS/intergenic/intergenic_data_{self.species}.csv.gz', index=False, compression = 'gzip')

		except FileNotFoundError:
			print(f"Error: The genomic fasta file for {self.species} was not found.")

		except PermissionError:
			print(f"Error: Permission denied to access the genomic fasta file for {self.species}.")

		except Exception as error:
			print(f"An unexpected error occurred while processing the genomic fasta file for {self.species}: {error}")





def process_species(species, group):
	print("\n\n")
	print(species)
	extractor = ExtractIntergenicSequences(species,
										   group,
										   nts_upstream_extract=500,
										   nts_downstream_extract=500)
	extractor.extract_intergenic_samples()

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