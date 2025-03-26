#!/usr/bin/env python3

import gzip
import re
import json
import os
import time
import pandas as pd
from itertools import combinations

class GFFProcesser:
	"""
	Process and extract relevant annotations from genomic GFF-files.
	Store in dict with each key corresponding to information about one mRNA transcript.
	"""

	def __init__(self, species):
		"""
        Initialize the GFFProcesser class.

        Args:
            species (str): The species' name.
        """

		self.species = species
		self.transcripts_info_dict = {}
		self.genes_info_dict = {}

	def extract_correct_positions(self, transcript_id):
		"""
		Ensure positions have been extracted correctly;
		if either exon positions or CDS positions are not annotated for mRNA transcript, then remove the transcript.

		Args:
			transcript_id (str): The particular ID of the transcript datapoint is extracted from. 
		"""

		#No annotated CDS (and thereby TIS)
		if len(self.transcripts_info_dict[transcript_id]["CDS_pos"]) == 0:
			del self.transcripts_info_dict[transcript_id]
			
		#No annotated exons (no defined mature mRNA)
		elif len(self.transcripts_info_dict[transcript_id]["exon_pos"]) == 0:
			del self.transcripts_info_dict[transcript_id]
		
		#Additional check if both CDS and exon positions are extracted	
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


	def extract_gene_positions(self):
		"""
		Extract names and positions of protein coding genes in gff-file and store in dict.
		"""

		#Initialize
		genes_ids = []

		#Open GFF-file with annotations
		with gzip.open(f"../../data/data_raw/genomic_gff/{self.species}_genomic.gff.gz", "rt") as info_file:
			#Read and discard the first 8 lines (information lines)
			for _ in range(8):
					next(info_file)

			#Continue reading file line by line
			for line in info_file:

				#Split information from columns
				col_info = line.split("\t")

				#Skip non-annotation lines
				if len(col_info) >= 3:

					#Extract annotations from all relevant sources
					if col_info[1] in ["BestRefSeq", "RefSeq", "Gnomon", "BestRefSeq%2CGnomon", "Curated Genomic", "RefSeqFE"]: 

						#Extract lines with protein-coding gene annotation, exclude trans-splicing annotations 
						#(trans-splicing annotations has more than one gene annotation, and are very rare (maximum found 5 instances per species)) 
						#gene_biotype=protein_coding ensures to only use protein coding genes
						if col_info[2] == "gene" and "gene_biotype=protein_coding" in line and "exception=trans-splicing" not in line:

							#Extract attribute information column
							attr_col = col_info[8]
							
							#Look for gene ID
							match_gene = re.search(r'ID=gene-([^;]+)', attr_col)	

							if match_gene:
								#Extract gene id, store name and gene positions in dict
								gene_id = match_gene.group(1).strip()

								#Store all identified gene IDs
								genes_ids.append(gene_id)

								#Add gene name and positions to dict
								self.genes_info_dict[gene_id] = [int(col_info[3]), int(col_info[4])]


		#Ensure that stored genes are only annotated once
		assert len(genes_ids) == len(list(set(genes_ids)))

		return self.genes_info_dict


	def process_gff_file(self):
		"""
		Process GFF-file line by line to extract complete annotations on mRNA-level.
		"""
		
		#Initialize
		prot_name = ""
		transcript_id = ""
		initial_mRNA_found = False
		prot_name_found = False
		genes_info_dict = self.extract_gene_positions()

		#Open GFF-file with annotations
		with gzip.open(f"../../data/data_raw/genomic_gff/{self.species}_genomic.gff.gz", "rt") as info_file:
			#Read and discard the first 8 lines (information lines)
			for _ in range(8):
					next(info_file)

			#Continue reading file line by line
			for line in info_file:

				#Split information from columns
				col_info = line.split("\t")

				#Skip non-annotation lines
				if len(col_info) >= 3:

					#Extract annotations from all relevant sources
					if col_info[1] in ["BestRefSeq", "RefSeq", "Gnomon", "BestRefSeq%2CGnomon", "Curated Genomic", "RefSeqFE"] and "pseudo=true" not in line: 

						#Extract lines with mRNA annotation
						if col_info[2] == "mRNA":

							#Make sure that positions have been extracted correctly
							if initial_mRNA_found and transcript_id in self.transcripts_info_dict.keys():
								self.extract_correct_positions(transcript_id)

							#Re-initialize for each new mRNA annotation
							prot_name = ""
							prot_name_found = False
							initial_mRNA_found = True

							#Extract attribute information column
							attr_col = col_info[8]

							#Look for transcript ID
							match_transcript = re.search(r'ID=rna-([^;]+)', attr_col)
							
							if match_transcript:
								#Extract transcript id
								transcript_id = match_transcript.group(1).strip()

								#Extract gene information giving rise to particular mRNA transcript
								match_gene = re.search(r'Parent=gene-([^;]+)', attr_col)

								if match_gene:
									gene_id = match_gene.group(1).strip()

									if gene_id in self.genes_info_dict.keys():

										#Write transcript information to dict and initialize
										self.transcripts_info_dict[transcript_id] = {"mRNA_pos": [int(col_info[3]), int(col_info[4])],		#mRNA coordinates on chromosome
																					 "Strand": col_info[6],									#Strand-tag
							                                                    	 "exon_pos": [],										#Annotated exon start- and stop coordinates
							                                                   		 "CDS_pos": [],											#Annotated CDS start- and stop coordinates
							                                               			 "chrom": col_info[0],									#Chromosome-tag
							                                               			 "source": col_info[1],									#Algorithm used for annotated feature
							                                               			 "gene": gene_id,										#Name of gene transcribed mRNA comes from
							                                               			 "gene_coordinates": genes_info_dict[gene_id]}			#Gene coordinates
								
						#Search for exon and CDS annotations
						if col_info[2] in ["exon", "CDS"]:
								
							#Extract attribute information column
							attr_col = col_info[8]

							#Look for exon- and CDS-annotations belonging to given transcript
							match_annotations = re.search(r'Parent=rna-([^;]+)', attr_col)
							if match_annotations:
								#Make sure annotated exon only belongs to one mRNA 
								#(the reason for doing this is that I have yet only discovered such cases from the gff-files. 
								#if this eventually was not the case, the code would need to account for exon annotations belonging 
								#to more than one transcript, as examplified on 
								#github https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md#parent-part_of-relationships)
								assert len(match_annotations.group(1).split(",")) == 1
									
								#Make sure the identified Parent mRNA ID belongs to the current mRNA
								if match_annotations.group(1) == transcript_id:

									if transcript_id in self.transcripts_info_dict.keys():

										#Extract exon positions
										if col_info[2] == "exon":
											self.transcripts_info_dict[transcript_id]["exon_pos"].append([int(col_info[3]), int(col_info[4])])

										#Extract CDS positions
										elif col_info[2] == "CDS":
											self.transcripts_info_dict[transcript_id]["CDS_pos"].append([int(col_info[3]), int(col_info[4])])

											#Check if one or more proteins arise from the mRNA transcript variant; 
											#from the gff-file structures I have not run into cases like this with the tested files, 
											#but the assertion makes sure we know if it happens and can modify to account for it 
											#(like if the same exon belongs to several mRNAs, example also from github)
											match_prot_name = re.search(r'Name=([^;]+);', attr_col)
												
											if match_prot_name:
												if not prot_name_found:
													#Store protein name from mRNA
													prot_name = match_prot_name.group(1)
												else: 
													assert prot_name == match_prot_name.group(1), "Transcript annotations corresponds to more than one protein."
												prot_name_found = True


		#Make sure that positions have been extracted correctly for last mRNA
		if transcript_id in self.transcripts_info_dict.keys():
			self.extract_correct_positions(transcript_id)

		#Print a summary from extraction
		print("Sequence annotations harvested from species", self.species, flush = True)
		print("Number of transcripts extracted:", len(self.transcripts_info_dict.keys()), flush = True)

		#Check for potential duplicate annotations
		for transcript_id in self.transcripts_info_dict:
			#Check that there are no duplicate exon position sets
			for (i, exon_coordinates1), (j, exon_coordinates2) in combinations(enumerate(self.transcripts_info_dict[transcript_id]["exon_pos"]), 2):
			    assert exon_coordinates1 != exon_coordinates2, "Duplicate exon annotation found."
				
				
			#Check that there are no duplicate CDS position sets
			for (i, CDS_coordinates1), (j, CDS_coordinates2) in combinations(enumerate(self.transcripts_info_dict[transcript_id]["CDS_pos"]), 2):
			    assert CDS_coordinates1 != CDS_coordinates2, "Duplicate CDS annotation found"

		#Save dict with annotation information
		with open(f"../../data/data_model_preparation/transcripts_info/{self.species}_dict.json", "w") as dict_file:
			json.dump(self.transcripts_info_dict, dict_file)


	def check_annotations(self):
		"""
		Do an additional check to ensure that annotations have been extracted properly.
		"""

		print("Conducting additional check on extracted annotations.")
		with open(f'../../data/data_model_preparation/transcripts_info/'+self.species+'_dict.json', 'r') as dict_file:
			annotations_dict = json.load(dict_file)

		for key, value in annotations_dict.items():
			try:
			    cds_positions = list(value.get('CDS_pos'))
			    unique_cds_positions = set(tuple(pos) for pos in cds_positions)

			    #Check that no CDS coordinate pair occurs more than once in a transcript.
			    assert len(cds_positions) == len(unique_cds_positions), "A CDS sequence occurs occurs twice for a transcript."

			    #Check that no start coordinate occurs twice in a transcript.
			    cds_start = []

			    for i in range(len(cds_positions)):
			    	cds_start.append(cds_positions[i][0])

			    assert len(cds_start) == len(set(cds_start)), "Something is wrong with a start coordinate of CDS."

			    #Check that no stop coordinate occurs twice in a transcript.
			    cds_end = []

			    for i in range(len(cds_positions)):
			    	cds_end.append(cds_positions[i][1])

			    assert len(cds_end) == len(set(cds_end)), "Something is wrong with a stop coordinate of CDS."
			
			except AssertionError:
				print(self.species)
				print(key, value)

		print("Check completed.\n")
		    

def main():
	"""Extract annotations from gff-files of each species"""

	#Extract list of all species names
	species_filenames_list = os.listdir("../../data/data_raw/genomic_gff")
	species_list = [filename.split("_genomic.gff.gz")[0] for filename in species_filenames_list]

	#Process annotations in gff-file for each species
	for species in species_list:
		process_information = GFFProcesser(species)

		#Main code
		process_information.process_gff_file()
		process_information.check_annotations()


start_time = time.time()
main()
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds", flush = True)
