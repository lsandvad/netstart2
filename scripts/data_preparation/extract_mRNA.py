import gzip
import re
import json
import time
import pandas as pd
import random
import multiprocessing


class ExtractSequences:
	"""
	Extract coding sequences and label TIS and non-TIS locations in mature mRNA sequences. 
	Specifically, the script extracts data for the (1) ATG TIS-labelled dataset, and 
	(2) the ATG non-TIS-labelled dataset from mature mRNA. 
	"""

	def __init__(self, species, group, nts_upstream_extract, nts_downstream_extract):
		"""
        Initialize the ExtractSequences class.

        Args:
            species (str): The species name.
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

				print("Transcripts from annotation file:", len(list(annotations_dict.keys())), flush = True)

				return annotations_dict
	
		except FileNotFoundError:
			print(f"Error: The annotation file for {self.species} was not found.", flush = True)

			return None

		except json.JSONDecodeError as error:
			print(f"Error decoding JSON for {self.species}: {error}", flush = True)

			return None

		except Exception as error:
			print(f"An unexpected error occurred for {self.species}: {error}", flush = True)

			return None

	def get_TIS_positions_gene(self):
		"""
		Get a list of all TISs from splice variants arising from each gene.
		"""

		#Initialize
		gene_TIS_positions = {}

		with open(f'../../data/data_model_preparation/transcripts_info/'+self.species+'_dict.json', 'r') as dict_file:
			annotations_dict = json.load(dict_file)

			#Collect all TIS positions from isoforms of each gene, store in dict
			for key, value in annotations_dict.items():
				gene_name = value.get('gene')
				strand = value.get('Strand')
				CDS = value.get("CDS_pos")

				#Template strand
				if strand == "+":
					TIS = CDS[0][0] 		#TIS is start coordinate of first CDS
				#Complement strand
				else: 
					TIS = CDS[0][-1] 		#TIS is end coordinate on first CDS (highest position, complement strand)
				
				#Initialize for given gene
				if gene_name not in gene_TIS_positions.keys():
					gene_TIS_positions[gene_name] = []
					gene_TIS_positions[gene_name].append(TIS)
				else:
					if TIS not in gene_TIS_positions[gene_name]:
						gene_TIS_positions[gene_name].append(TIS)

		return gene_TIS_positions


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


	def quality_check_extracted_sequences(self,
									  TIS,
									  CDS_end,
									  exon_seq, 
									  cds_ends_correct, 
									  cds_ends_wrong,
									  multiple_3_correct,
									  multiple_3_wrong,
									  stop_codon_in_cds,
									  strand,
									  genomic_ATG_coord,
									  genomic_ATG_coord_correct,
									  genomic_ATG_coord_false,
									  chromosome_seq):
		"""
		Perform several quality checks of the extracted sequences.

		Args:
			TIS: Annotated position of TIS 
			CDS_end: Annotated end position of CDS
			exon_seq: The exon sequence
			cds_ends_correct: counter for the number of extracted CDS sequences that ends with a stop codon
			cds_ends_wrong: counter for the number of extracted CDS sequences that does not end with a stop codon
			multiple_3_correct: counter for the number of CDSs corresponding to a complete number of triplets
			multiple_3_wrong: counter for the number of CDSs corresponding to an incomplete number of triplets
			stop_codon_in_cds: check for in-frame stop codons in CDS
			strand: the strand that the CDS is placed on (template/complement)
			genomic_ATG_coord: the genomic coordinates of the ATG
			genomic_ATG_coord_correct: counter for the sequences where the genomic ATG coordinates are correct
			genomic_ATG_coord_false: counter for the sequences where the genomic ATG coordinates are not correct
			chromosome_seq: the assembly ID (chromosome, scaffold, etc.)
		"""
	
		write_transcript_annotation = True

		#CDS end position does not correspond to position in exon; something is wrong with annotation
		if CDS_end == "":
			write_transcript_annotation = False
		else:
			len_protein_coding_part = len(exon_seq[TIS:CDS_end])


		if write_transcript_annotation:
			#Check that coding sequence corresponds to a number of triplets
			if len_protein_coding_part % 3 != 0: 
				write_transcript_annotation = False									#Skip annotations with errors
				multiple_3_wrong += 1
			else: 
				multiple_3_correct += 1
									
			#Ensure that the extracted CDS ends with a stop codon (if not, there is an error in annotation)
			if exon_seq[CDS_end-3:CDS_end].lower() in ["tga", "tag", "taa"]:
				cds_ends_correct += 1
			else:
				cds_ends_wrong += 1
				write_transcript_annotation = False									#Skip annotations with errors

			codons_in_seq = int(len_protein_coding_part/3) - 1

			#Check for in-frame stop codons in sequence
			for i in range(codons_in_seq):
				if exon_seq[TIS+i*3:TIS+i*3+3].lower() in ["tga", "tag", "taa"]:
					write_transcript_annotation = False
					stop_codon_in_cds += 1
					break

			if strand == "+":
				if chromosome_seq[genomic_ATG_coord:genomic_ATG_coord+3].upper() != "ATG": #ATG (genomic coordinates)
					write_transcript_annotation = False
					genomic_ATG_coord_false += 1
				else:
					genomic_ATG_coord_correct += 1

			elif strand == "-":
				if chromosome_seq[genomic_ATG_coord:genomic_ATG_coord+3].upper() != "CAT": #reverse-complementet ATG (corresponds to genomic coordinates)
					write_transcript_annotation = False
					genomic_ATG_coord_false += 1
				else:
					genomic_ATG_coord_correct += 1

		return write_transcript_annotation, cds_ends_correct, cds_ends_wrong, multiple_3_correct, multiple_3_wrong, \
			   stop_codon_in_cds, genomic_ATG_coord_correct, genomic_ATG_coord_false

	def extract_non_TIS_samples(self,
								annotations_dict, 
								transcript_id, 
								exon_seq,
								TIS,
								all_TISs,
								non_TIS_information,
								len_5_UTR):
		"""
		Extract sequences and informative additional data for non-TIS ATG labelled datapoints in mature mRNA.

		Args:
			annotation_dict (dict): Dict with annotations, processed from gff-file.
			transcript_id (str): The particular ID of the transcript, datapoint is extracted from. 
			exon_seq (str): The mature mRNA sequence. 
			TIS (int): The first position of a start codon TIS.
			all_TIS (list): Positions of all TIS in a given gene (arising from isoforms).
			non_TIS_information (df): dataframe containing the information in the final dataset-format. 
			len_5_UTR (int): The length of the 5' UTR if annotated
		"""
		
		try: 
			#Handle negative cases (ATG present, but not the TIS)
			atg_indices = [i for i in range(len(exon_seq)) if exon_seq.upper().startswith("ATG", i)]

			assert TIS in atg_indices, "TIS not found in ATG indices"

			#Remove all ATG TIS positions from list (arising from potential different isoforms of same gene with distinct TISs.
			#We want to make sure that no ATg TIS is falsely labelled as non-TIS).
			atg_indices = [ATG for ATG in atg_indices if ATG not in all_TISs]

			#Extract ATG indices upstream TIS
			indices_upstream_TIS = [index for index in atg_indices if index < TIS and index > self.nts_upstream_extract]
			indices_upstream = len(indices_upstream_TIS)

			#####Handle upstream ATG cases#####

			#Only include data from upstream ATGs in cases where 5' UTR length is known (and not estimated)
			if len_5_UTR is not None: 
				for i in range(len(indices_upstream_TIS)):

					#Calculate relative position of ATG to TIS, -1 for closest, -2 for next closest, and so on
					ATG_pos = -(indices_upstream - i) 
									    
					if (TIS - indices_upstream_TIS[i]) % 3 == 0:
						codon_type = "Upstream, in frame"
					else:
						codon_type = "Upstream, out of frame"

					#Make sure that an ATG has been extracted as the label (and not any other codon)
					assert exon_seq[indices_upstream_TIS[i]:indices_upstream_TIS[i]+3].upper() == "ATG", \
						   "ATG-label not extracted properly."

					#Write correctly-labeled/true datapoint to dataframe
					non_TIS_information = pd.concat([non_TIS_information, 
													pd.DataFrame({'Sequence': exon_seq.upper()[indices_upstream_TIS[i]-self.nts_upstream_extract:\
																							   indices_upstream_TIS[i]+3+self.nts_downstream_extract],			#Extracted sequence
																  'ATG_position': self.nts_upstream_extract,													#Position of ATG to predict on
																  'TIS': 0,																						#FALSE
																  'codon_type': codon_type,																		#Upstream/downstream, in or out of frame
																  'stop_codon': None,																			#None
																  'species': self.species,																		#Species
																  'group': self.group,																			#Eukaryotic grouping system
																  'annotation_source': [annotations_dict[transcript_id]["source"]],								#Source of annotation (Gnomon, RefSeq, BestRefSeq)
																  '5_UTR_length_annotated': None,																#Length of 5' UTR
																  'ATG_relative': ATG_pos,																		#Position of the ATG; 0 implies a TIS, -X implies upstream, X implies downstream
																  'landmark_id': [annotations_dict[transcript_id]["chrom"]],									#Chromosome
																  'mrna_pos': [annotations_dict[transcript_id]["mRNA_pos"]],									#The transcript coordinates
																  'transcript_id': transcript_id,																#transcript id
																  'strand': [annotations_dict[transcript_id]["Strand"]],										#transcript strand
																  'ATG_genomic_position': None,																	#The TIS position (from genomic data)
																  'gene': [annotations_dict[transcript_id]["gene"]],											#gene name
																  'gene_coordinates': [annotations_dict[transcript_id]["gene_coordinates"]]})], 				#gene coordinates													
													ignore_index=True)

			
			
			#####Handle downstream ATG cases#####
			#Extract ATG indices downstream TIS
			indices_downstream_TIS = [index for index in atg_indices if index >= TIS]

			#Shuffle the list of indices downstream TIS
			random.shuffle(indices_downstream_TIS)

			sorted_downstream = sorted(indices_downstream_TIS)

			#Initialize (store one sample downstream TIS in frame and out of frame, respectively)
			in_frame_search = True
			out_of_frame_search = True
			count_in_frame = 0
								
			for atg_index in indices_downstream_TIS: 
				#Determine reading frame relative to TIS
				if in_frame_search: 
					if (atg_index - TIS) % 3 == 0:
						#Only use non-coding sequences with X nts upstream and Y nts downstream an ATG
						if atg_index > self.nts_upstream_extract and len(exon_seq) - (atg_index + 3) > self.nts_downstream_extract:
												
							#Define relative position of ATG to TIS
							ATG_pos = sorted_downstream.index(atg_index) + 1
							#Assign codon type
							codon_type = "Downstream, in frame"

							#Write correctly-labeled/true datapoint to dataframe
							non_TIS_information = pd.concat([non_TIS_information, 
															pd.DataFrame({'Sequence': exon_seq.upper()[atg_index-self.nts_upstream_extract:\
																									   atg_index+3+self.nts_downstream_extract],		#Extracted sequence
																		  'ATG_position': self.nts_upstream_extract,									#Position of ATG to predict on
																		  'TIS': 0,																		#FALSE
																		  'codon_type': codon_type,														#upstream/downstream, in or out of frame
																		  'stop_codon': None,															#None
																		  'species': self.species,														#Species
																		  'group': self.group,															#Eukaryotic grouping system
																		  'annotation_source': [annotations_dict[transcript_id]["source"]],				#Source of annotation (Gnomon, RefSeq, BestRefSeq)
																		  '5_UTR_length_annotated': None,												#Length of 5' UTR
																		  'ATG_relative': ATG_pos,														#Position of the ATG; 0 implies a TIS, -X implies upstream, X implies downstream
																		  'landmark_id': [annotations_dict[transcript_id]["chrom"]],					#Chromosome
																		  'mrna_pos': [annotations_dict[transcript_id]["mRNA_pos"]],					#The transcript coordinates
																		  'transcript_id': transcript_id,												#transcript id
																		  'strand': [annotations_dict[transcript_id]["Strand"]],						#transcript strand
																  		  'ATG_genomic_position': None,													#The TIS position (from genomic data)
																  		  'gene': [annotations_dict[transcript_id]["gene"]],							#gene name
																  		  'gene_coordinates': [annotations_dict[transcript_id]["gene_coordinates"]]})], #gene coordinates			
															ignore_index=True)
							
							count_in_frame += 1

							if count_in_frame == 2:
								#Only extract 2 downstream ATG in frame
								in_frame_search = False
									
				if out_of_frame_search:
					if (atg_index - TIS) % 3 != 0:
						#Only use non-coding sequences with X nts upstream and Y nts downstream an ATG
						if atg_index > self.nts_upstream_extract and len(exon_seq) - (atg_index + 3) > self.nts_downstream_extract:
												
							#Define relative position of ATG to TIS
							ATG_pos = sorted_downstream.index(atg_index) + 1
												
							#Assign codon type
							codon_type = "Downstream, out of frame"

							#Write correctly-labeled/true datapoint to dataframe
							non_TIS_information = pd.concat([non_TIS_information, 
															pd.DataFrame({'Sequence': exon_seq.upper()[atg_index-self.nts_upstream_extract:\
																									   atg_index+3+self.nts_downstream_extract],		#Extracted sequence
																		  'ATG_position': self.nts_upstream_extract,									#Position of ATG to predict on
																		  'TIS': 0,																		#FALSE
																		  'codon_type': codon_type,														#upstream/downstream, in or out of frame
																		  'stop_codon': None,															#None
																		  'species': self.species,														#Species
																		  'group': self.group,															#Eukaryotic grouping system
																		  'annotation_source': [annotations_dict[transcript_id]["source"]],				#Source of annotation (Gnomon, RefSeq, BestRefSeq)
																		  '5_UTR_length_annotated': None,												#Length of 5' UTR
																		  'ATG_relative': ATG_pos,														#Position of the ATG; 0 implies a TIS, -X implies upstream, X implies downstream
																		  'landmark_id': [annotations_dict[transcript_id]["chrom"]],					#Chromosome
																		  'mrna_pos': [annotations_dict[transcript_id]["mRNA_pos"]],					#The transcript coordinates
																		  'transcript_id': transcript_id,												#transcript id
																		  'strand': [annotations_dict[transcript_id]["Strand"]],						#transcript strand
																  		  'ATG_genomic_position': None,													#The TIS position (from genomic data)
																  		  'gene': [annotations_dict[transcript_id]["gene"]],							#gene name
																  		  'gene_coordinates': [annotations_dict[transcript_id]["gene_coordinates"]]})], #gene coordinates			
															ignore_index=True)
							
							#Only extract 1 downstream ATG out of frame
							out_of_frame_search = False

			return non_TIS_information

		except AssertionError as assertion_error:
			print(f"Assertion error for {self.species}: {assertion_error}", flush = True)

		except Exception as error:
			print(f"An unexpected error occurred for {self.species}: {error}", flush = True)

	def extract_strand_information(self, 
								   annotations_dict, 
								   transcript_id, 
								   chromosome_seq, 
								   TIS_information, 
								   non_TIS_information,
								   cds_ends_correct, 
								   cds_ends_wrong,
								   multiple_3_correct,
								   multiple_3_wrong,
								   stop_codon_in_cds,
								   gene_TIS_positions_dict,
								   genomic_ATG_coord_correct,
								   genomic_ATG_coord_false):
		
		"""
		Extract sequence information on the respective strand of an mRNA annotation. 
		The function is run for every individual transcript.

		Args:
			annotation_dict (dict): Dict with annotations, processed from gff-file.
			transcript_id (str): The particular ID of the transcript, datapoint is extracted from. 
			chromosome_seq (str): The sequence of a landmark (scaffold, chromosome, etc. dependent on assembly).
			TIS_information (df): dataframe containing the information in the final dataset-format, for the TIS labelled ATG samples.
			non_TIS_information (df): dataframe containing the information in the final dataset-format, for the non-TIS labelled ATG samples.
			cds_ends_correct (int): number of transcripts in which the coding sequence ends with an actual stop codon. 
			cds_ends_wrong (int): number of transcripts in which the coding sequence ends with another codon than a stop codon. 
			multiple_3_correct (int): number of coding sequences of transcripts containing a multiple of codon triplets.
			multiple_3_correct (int): number of coding sequences of transcripts not containing a multiple of codon triplets.
			stop_codon_in_cds (int): number of transcripts with a in-frame stop codon within CDS.
			gene_TIS_positions_dict (dict): dict with all genomic TIS positions of a given gene (due to isoforms).
			genomic_ATG_coord_correct: counter for the sequences where the genomic ATG coordinates are correct.
			genomic_ATG_coord_false: counter for the sequences where the genomic ATG coordinates are not correct.
		"""
		
		#Initialize for every transcript
		exon_seq_list = []
		exons_length = 0
		CDS_end = ""
		pattern = re.compile("[^ACGTagct]")
		all_TISs = []

		#get all TIS positions in gene
		gene = annotations_dict[transcript_id]["gene"]
		all_TIS_positions_gene = gene_TIS_positions_dict[gene]

		#Handle cases on template strand
		if annotations_dict[transcript_id]["Strand"] == "+": 

			strand = "+"

			#Sort lists of exon and CDS positions based on start position of each exon and CDS, respectively
			sorted_exons = sorted(annotations_dict[transcript_id]["exon_pos"], key=lambda x: x[0])
			sorted_CDS = sorted(annotations_dict[transcript_id]["CDS_pos"], key=lambda x: x[0])

			#Locate start position (TIS) and end position (stop codon) of CDS
			CDS_start = sorted_CDS[0][0]
			CDS_stop = sorted_CDS[-1][-1]
			exon_start = sorted_exons[0][0]

			genomic_ATG_coord = CDS_start-1
			
			#Loop over every exon-region in mRNA
			for exon_pos in sorted_exons:

				if exon_pos[0] <= CDS_start <= exon_pos[1]:
					#Locate TIS position in the mature mRNA (introns removed)
					TIS = exons_length + CDS_start - exon_pos[0]						#TIS: 0-index

				if exon_pos[0] <= CDS_stop <= exon_pos[1]:
					#Locate stop codon/end of CDS in the mature mRNA (introns removed)
					CDS_end = exons_length + CDS_stop - exon_pos[0] + 1					#CDS_end: 0-index

				#Store all possible TIS from gene due to isoforms (store positions in transcript)
				for TIS_position in all_TIS_positions_gene:
					if exon_pos[0] <= TIS_position <= exon_pos[1]:
						#Locates positions in the mature mRNA (introns removed)
						all_TISs.append(exons_length + TIS_position - exon_pos[0])		#TIS: 0-index

				exons_length += exon_pos[1] - exon_pos[0] + 1 							#Total length of exons in mRNA				
				exon_seq_list.append(chromosome_seq[exon_pos[0]-1:exon_pos[1]])			#The total exon sequence

			#Join list of exon chunks to full exon sequence
			exon_seq = "".join(exon_seq_list)

			#Conduct quality check on extracted sequence
			write_transcript_annotation, cds_ends_correct, cds_ends_wrong, multiple_3_correct, multiple_3_wrong, stop_codon_in_cds, \
				genomic_ATG_coord_correct, genomic_ATG_coord_false = self.quality_check_extracted_sequences(TIS,
																											CDS_end,
																											exon_seq, 
																											cds_ends_correct, 
																											cds_ends_wrong,
																											multiple_3_correct,
																											multiple_3_wrong,
																											stop_codon_in_cds,
																											strand,
																											genomic_ATG_coord,
																											genomic_ATG_coord_correct,
																											genomic_ATG_coord_false,
																											chromosome_seq)


			#Adjust extracted sequence in cases where 5' UTR length is not annotated; add 300 nucleotides upstream and note estimated 5' UTR
			if TIS == 0:
				exon_seq = chromosome_seq[(CDS_start - 1) - 300:CDS_start-1] + exon_seq
				TIS = 300
				add_seq = 300
				all_TISs = [TIS + 300 for TIS in all_TISs] #Also adjust all possible TIS
				len_5_UTR = None
			else:
				len_5_UTR = TIS - 1
				add_seq = 0

				#When the annotated 5' UTR is shorter then 300 nucleotides, add upstream genome sequence
				if TIS < 300:
					add_seq = 300 - TIS
					exon_seq = chromosome_seq[(exon_start - 1) - add_seq:exon_start-1] + exon_seq
					TIS = TIS + add_seq
					all_TISs = [TIS + add_seq for TIS in all_TISs] #Also adjust all possible TIS


		#Handle cases on complement strand
		if annotations_dict[transcript_id]["Strand"] == "-":

			strand = "-"

			#########################
			##Extract exon sequence##
			#########################
			#Sort lists of exon and CDS positions based on start position of each exon and CDS, respectively
			sorted_exons = sorted(annotations_dict[transcript_id]["exon_pos"], key=lambda x: x[0])
			sorted_CDS = sorted(annotations_dict[transcript_id]["CDS_pos"], key=lambda x: x[0])

			#Extract 300 nucleotides upstream TIS (use for cases where 5' UTR length is not annotated)
			UTR_5_seq = chromosome_seq[sorted_exons[-1][-1]:sorted_exons[-1][-1] + 300]
								
			for exon_pos in sorted_exons:
				exon_seq_list.append(chromosome_seq[exon_pos[0]-1:exon_pos[1]])					#The total exon sequence seen from the template strand

			exon_seq = ''.join(exon_seq_list)

			exons_and_5_utr_seq = exon_seq + UTR_5_seq
			#Get complement strand sequence by taking reverse complement of sequence
			exon_seq = self.reverse_complement(exon_seq)
			exons_and_5_utr_seq = self.reverse_complement(exons_and_5_utr_seq)

			#########################
			#######Extract TIS#######
			#########################
			#Sort lists of exon and CDS positions based on start position of each exon and CDS, respectively, but reversed
			sorted_exons = sorted(annotations_dict[transcript_id]["exon_pos"], key=lambda x: x[0], reverse=True)
			sorted_CDS = sorted(annotations_dict[transcript_id]["CDS_pos"], key=lambda x: x[0], reverse = True)
			
			#Locate start position (TIS) and end position (stop codon) of CDS corresponding to the ones on complement strand	
			CDS_start = sorted_CDS[0][-1]
			CDS_stop = sorted_CDS[-1][0]

			genomic_ATG_coord = CDS_start-3 #minus

			for exon_pos in sorted_exons:
				if exon_pos[0] <= CDS_start <= exon_pos[1]:
					#Locate TIS position in the mature mRNA (introns removed)
					TIS = exons_length + exon_pos[1] - CDS_start					#TIS; 0-index
				
				if exon_pos[0] <= CDS_stop <= exon_pos[1]:
					#Locate stop codon/end of CDS in the mature mRNA (introns removed)
					CDS_end = exons_length + exon_pos[1] - CDS_stop	+ 1	

				#Store all possible TIS from gene due to isoforms (store positions in transcript)
				for TIS_position in all_TIS_positions_gene:
					if exon_pos[0] <= TIS_position <= exon_pos[1]:
						#Locate TIS positions in the mature mRNA (introns removed)
						all_TISs.append(exons_length + exon_pos[1] - TIS_position)	#TIS: 0-index	

				exons_length += exon_pos[1] - exon_pos[0] + 1 						#Total length of exons in mRNA


			#Conduct quality check on extracted sequence
			write_transcript_annotation, cds_ends_correct, cds_ends_wrong, multiple_3_correct, multiple_3_wrong, stop_codon_in_cds, \
				genomic_ATG_coord_correct, genomic_ATG_coord_false = self.quality_check_extracted_sequences(TIS,
																											CDS_end,
																											exon_seq, 
																											cds_ends_correct, 
																											cds_ends_wrong,
																											multiple_3_correct,
																											multiple_3_wrong,
																											stop_codon_in_cds,
																											strand,
																											genomic_ATG_coord,
																											genomic_ATG_coord_correct,
																											genomic_ATG_coord_false,
																											chromosome_seq)

			#Adjust extracted sequence in cases where 5' UTR length is not annotated
			if TIS == 0:
				exon_seq = exons_and_5_utr_seq
				TIS = 300
				add_seq = 300
				all_TISs = [TIS + 300 for TIS in all_TISs] 							#Also adjust all possible TIS
				len_5_UTR = None
			else:
				len_5_UTR = TIS - 1
				add_seq = 0

				if TIS < 300:
					add_seq = 300 - TIS
					exon_seq = exons_and_5_utr_seq[add_seq:]
					TIS = TIS + add_seq
					all_TISs = [TIS + add_seq for TIS in all_TISs] 					#Also adjust all possible TIS


		#Use regular expression to find any characters other than A, C, G, or T
		result = pattern.search(exon_seq)
			
		#Make sure that sequence was correctly assigned (seq % 3 = 0 AND correct stop codon at stop_codon)
		if write_transcript_annotation:
			#Only use sequences without missing nucleotide descriptions
			if not result:
				#Only extract ATG TIS data
				if exon_seq[TIS:TIS+3] == "ATG":
					TIS_information = pd.concat([TIS_information, pd.DataFrame({'Sequence': exon_seq.upper(),													#Extracted sequence
																				'ATG_position': TIS,															#Position of ATG to predict on
																				'TIS': 1,																		#TRUE
																				'codon_type': "TIS",															#"TIS"
																				'stop_codon': CDS_end + add_seq,												#Last position of protein coding region
																				'species': self.species,														#species
																				'group': self.group,															#eukaryotic grouping; for now, None
																				'annotation_source': [annotations_dict[transcript_id]["source"]],				#Source of annotation (Gnomon, RefSeq, BestRefSeq)
																				'5_UTR_length_annotated': len_5_UTR,											#Length of 5' UTR
																				'ATG_relative': 0,																#Position of the ATG; 0 implies a TIS, -X implies upstream, X implies downstream
																				'landmark_id': [annotations_dict[transcript_id]["chrom"]],						#Chromosome
																				'mrna_pos': [annotations_dict[transcript_id]["mRNA_pos"]],						#The transcript coordinates
																				'transcript_id': transcript_id,													#transcript id
																				'strand': [annotations_dict[transcript_id]["Strand"]],							#transcript strand
																			  	'ATG_genomic_position': [[genomic_ATG_coord,genomic_ATG_coord+3]],				#The TIS position (from genomic data)
																			  	'gene': [annotations_dict[transcript_id]["gene"]],								#gene name
																			  	'gene_coordinates': [annotations_dict[transcript_id]["gene_coordinates"]]})], 	#gene coordinates		
												ignore_index=True)

					#Extract datapoints of non-TIS ATGs
					non_TIS_information = self.extract_non_TIS_samples(annotations_dict = annotations_dict, 
						                                               transcript_id = transcript_id, 
						                                               exon_seq = exon_seq,
						                                               TIS = TIS,
						                                               all_TISs = all_TISs,
						                                               non_TIS_information = non_TIS_information,
						                                               len_5_UTR = len_5_UTR)

		return TIS_information, non_TIS_information, cds_ends_correct, cds_ends_wrong, multiple_3_correct, multiple_3_wrong, \
			   stop_codon_in_cds, genomic_ATG_coord_correct, genomic_ATG_coord_false
							

	def process_and_extract_sequences(self):
		"""
		Process and extract mRNA sequence information from an entire genomic fasta file,
		based on annotations extracted from corresponding gff-file
		"""

		try: 
			#Load dict with transcript annotations
			annotations_dict = self.load_annotation_info_dict()
			gene_TIS_positions_dict = self.get_TIS_positions_gene()

			#Initialize
			chromosome_seq_initialized = False
			extract_info = True
			cds_ends_correct = 0
			cds_ends_wrong = 0
			multiple_3_correct = 0
			multiple_3_wrong = 0
			stop_codon_in_cds = 0
			genomic_ATG_coord_false = 0
			genomic_ATG_coord_correct = 0
			counter = 0
			transcripts_no = len(annotations_dict)

			#Create empty DataFrame with column names for positive samples (TIS ATGs)
			TIS_information = pd.DataFrame(columns=['Sequence', 					#extracted mature mRNA
													'ATG_position',		 			#Position of ATG to predict on
													'TIS',							#TRUE (1)
													'codon_type',					#"TIS"
													'stop_codon',					#stop codon position in sequence
													'species',						#species
													'group',						#eukaryotic grouping system
													'annotation_source',			#Gnomon or RefSeq
													'5_UTR_length_annotated',		#The 5' UTR (if annotated)
													'ATG_relative',					#position of ATG relative to TIS (0)
													'landmark_id',					#landmark ID given sequence was found on
													'mrna_pos',						#mrna coordinate of sequence
													'transcript_id',				#transcript ID
													'strand',						#strand
													'ATG_genomic_position',			#The TIS position (from genomic data)
													'gene',							#gene sequence originates from
													'gene_coordinates'])			#genomic coordinates of transcript that sequence originates from

			#Create empty DataFrame with column names for negative samples (non-TIS ATGs)
			non_TIS_information = pd.DataFrame(columns=['Sequence', 				#extracted mature mRNA window
														'ATG_position',		 		#Position of ATG to predict on
														'TIS',						#FALSE (0)
														'codon_type',				#upstream, downstream, in frame, out of frame
														'stop_codon',				#None
														'species',					#species
														'group',					#eukaryotic grouping system
														'annotation_source',		#Gnomon or RefSeq
														'5_UTR_length_annotated',	#None
														'ATG_relative',				#position of ATG relative to TIS (0)
														'landmark_id',				#landmark ID given sequence was found on
														'mrna_pos',					#mrna coordinate of sequence
														'transcript_id',			#transcript ID
														'strand',					#strand
														'ATG_genomic_position',		#The TIS position (from genomic data) (None)
														'gene',						#gene sequence originates from
														'gene_coordinates'])		#genomic coordinates of transcript that sequence originates from

			#Open genome sequence in fasta format
			with gzip.open(f"../../data/data_raw/genomic_fna/{self.species}_genomic.fna.gz", "rt") as seq_file:
				
				#Iterate line by line
				for line in seq_file:

					#Every time a new chromosome starts
					if line.startswith(">"):

						if chromosome_seq_initialized:

							chromosome_seq = ''.join(chromosome_seq_parts)

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

									counter += 1

									#Extract sample data from transcript
									TIS_information, \
									non_TIS_information, \
									cds_ends_correct, \
									cds_ends_wrong, \
									multiple_3_correct, \
									multiple_3_wrong, \
									stop_codon_in_cds, \
									genomic_ATG_coord_correct, \
									genomic_ATG_coord_false = self.extract_strand_information(annotations_dict = annotations_dict, 
																					   transcript_id = transcripts_in_chromosome[i], 
																					   chromosome_seq = chromosome_seq, 
																					   TIS_information = TIS_information,
																					   non_TIS_information = non_TIS_information,
																					   cds_ends_correct = cds_ends_correct,
																					   cds_ends_wrong = cds_ends_wrong,
																					   multiple_3_correct = multiple_3_correct,
																					   multiple_3_wrong = multiple_3_wrong,
																					   stop_codon_in_cds = stop_codon_in_cds,
																					   gene_TIS_positions_dict = gene_TIS_positions_dict,
																					   genomic_ATG_coord_correct = genomic_ATG_coord_correct,
																					   genomic_ATG_coord_false = genomic_ATG_coord_false)
									if counter % 500 == 0:
										print("Processed ", counter, "/", transcripts_no, "transcripts in species ", self.species, flush = True)

						#Store chromosome annotation for mapping transcripts to correct chromosome
						chrom_tag = line.split(" ")[0].strip(">")
						
						#Only store nuclear sequences
						if "mitochondrion" in line or "chloroplast" in line:
							extract_info = False
						else:
							extract_info = True

						#Re-initialize
						chromosome_seq_initialized = True
						chromosome_seq_parts = []


					#Every sequence line belonging to current chromosome is stored
					else: 
						if extract_info:
							chromosome_seq_parts.append(line.strip())


				#For the last iteration (the last landmark/chromosome identifier)
				#Only extract from nuclear genes
				print("Last Landmark reached.", flush = True)
				if extract_info:

					#Initialize on new for every chromosome
					transcripts_in_chromosome = []
					chromosome_seq = ''.join(chromosome_seq_parts)

					#Extract all transcripts on currently stored chromosome
					for transcript_id, transcript_info in annotations_dict.items():
						if 'chrom' in transcript_info and transcript_info['chrom'] == chrom_tag:
							transcripts_in_chromosome.append(transcript_id)

					#Loop over last transcript seperately to extract information individually
					for i in range(len(transcripts_in_chromosome)):

						counter += 1

						#Extract sample data from transcript
						TIS_information, \
						non_TIS_information, \
						cds_ends_correct, \
						cds_ends_wrong, \
						multiple_3_correct, \
						multiple_3_wrong, \
						stop_codon_in_cds, \
						genomic_ATG_coord_correct, \
						genomic_ATG_coord_false = self.extract_strand_information(annotations_dict = annotations_dict, 
																				  transcript_id = transcripts_in_chromosome[i], 
																				  chromosome_seq = chromosome_seq, 
																				  TIS_information = TIS_information,
																				  non_TIS_information = non_TIS_information,
																				  cds_ends_correct = cds_ends_correct,
																				  cds_ends_wrong = cds_ends_wrong,
																				  multiple_3_correct = multiple_3_correct,
																				  multiple_3_wrong = multiple_3_wrong,
																				  stop_codon_in_cds = stop_codon_in_cds,
																				  gene_TIS_positions_dict = gene_TIS_positions_dict,
																				  genomic_ATG_coord_correct = genomic_ATG_coord_correct,
																				  genomic_ATG_coord_false = genomic_ATG_coord_false)
						if counter % 500 == 0:
							print("Processed ", counter, "/", transcripts_no, "transcripts in species ", self.species, flush = True)

				print(self.species, "Percentage correct stop codons: ", \
					str(cds_ends_correct/(cds_ends_correct+cds_ends_wrong) * 100), \
					" (", str(cds_ends_correct), "/", str(cds_ends_correct+cds_ends_wrong), ")", flush = True)
				
				print(self.species, "Percentage correct codon sequences (% 3 = 0): ", \
					str(multiple_3_correct/(multiple_3_correct+multiple_3_wrong) * 100), \
					" (", str(multiple_3_correct), "/", str(multiple_3_correct+multiple_3_wrong), ")", flush = True)

				print(self.species, "Percentage of correct genomic placements (ATG start codons placed at genomic TIS coordinates):", \
					str(genomic_ATG_coord_correct/(genomic_ATG_coord_correct+genomic_ATG_coord_false) * 100), \
					" (", str(genomic_ATG_coord_correct), "/", str(genomic_ATG_coord_correct+genomic_ATG_coord_false), ")", flush = True)

				print(self.species, "Transcripts without an in-frame stop codon in CDS: ", str(100-(stop_codon_in_cds/(cds_ends_correct+cds_ends_wrong))*100), "%", flush = True)

				#For TIS_information DataFrame
				TIS_information['Seq_number'] = ['TIS_seq_{}'.format(i) for i in range(1, len(TIS_information) + 1)]

				#For non_TIS_information DataFrame
				non_TIS_information['Seq_number'] = ['mRNA_non_TIS_seq_{}'.format(i) for i in range(1, len(non_TIS_information) + 1)]

				#Save True-label coding dataset to CSV
				TIS_information.to_csv(f'../../data/data_model_preparation/datasets/TIS/mRNA_positive_{self.species}.csv.gz', index=False, compression = "gzip")

				#Save False-label coding dataset to CSV
				non_TIS_information.to_csv(f'../../data/data_model_preparation/datasets/non_TIS/mRNA/mRNA_negative_{self.species}.csv.gz', index=False, compression = "gzip")

		except FileNotFoundError:
			print(f"Error: The genomic fasta file for {self.species} was not found.", flush = True)

		except PermissionError:
			print(f"Error: Permission denied to access the {self.species} genomic fasta file.", flush = True)

		except Exception as error:
			print(f"An unexpected error occurred while processing the species {self.species} genomic fasta file: {error}", flush = True)


def process_species(species, group):
	print("\n\n", flush = True)
	print(species, flush = True)
	extractor = ExtractSequences(species,
							     group,
								 nts_upstream_extract = 500,
								 nts_downstream_extract = 500)
	extractor.process_and_extract_sequences()

def main():
	"""
	Run pipeline for several species.
	"""

	#Extract dict with species and belonging group
	#Read the CSV file into a DataFrame
	df_species = pd.read_csv("../../data/data_raw/species_information/species_groups.csv")

	#Create a dictionary from the DataFrame with "Species" as keys and "Group" as values
	species_dict = df_species.set_index('Species')['Group'].to_dict()

	#run lines if running non-parallel for tests
	for species, group in species_dict.items():
		process_species(species, group)

	"""
	#Create a multiprocessing Pool with the number of processes you want to run concurrently
	num_processes = multiprocessing.cpu_count()  # You can adjust this as needed
	pool = multiprocessing.Pool(processes=num_processes)

	#Use multiprocessing to process species concurrently
	for species, group in species_dict.items():
		pool.apply_async(process_species, args=(species, group))

	# Close the pool and wait for all processes to finish
	pool.close()
	pool.join()
	"""


start_time = time.time()
main()
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds", flush = True)
