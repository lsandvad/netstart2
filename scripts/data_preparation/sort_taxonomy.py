#!/usr/bin/env python3

import os
import zipfile
import io
import pandas as pd

#Requires: 
	#nodes.dmp (collected from NCBIs taxonomy database)
	#names.dmp (collected from NCBIs taxonomy database)
def sort_taxonomy_eukaryota():
	"""
	Creates a file with all eukaryotic organisms' taxonomical distribution across several ranks.
	"""

	#Open input taxonomy information directory
	with zipfile.ZipFile('../../data/data_raw/taxonomy_information/taxdmp.zip', 'r') as zip_file:
		
		#Open file with taxonomy nodes data
		with zip_file.open('nodes.dmp', 'r') as file:
			nodes = io.TextIOWrapper(file, encoding='utf-8')
        
			#Initialize
			nodes_dict = dict()

			#Extract all nodes' relevant information in dict (information across all ranks)
			for line in nodes:
				node_line = line.split("\t")
				
				#dict content: 
					#key: Node ID
					#value: [Parent node ID, Rank]
				nodes_dict[node_line[0]] = [node_line[2], node_line[4]]

			#Close file
			nodes.close()
			print("Preparation of nodes-list done")

		#Open file with taxonomy names data
		with zip_file.open('names.dmp', 'r') as file:
			names = io.TextIOWrapper(file, encoding='utf-8')

			#Initialize
			names_dict = dict()

			#Store node Id and scientific name in dict
			for line in names:
				name_line = line.split("\t")

				if name_line[6] == "scientific name":
					#dict content: 
						#key: Node ID, 
						#value: Scientific name
					names_dict[name_line[0]] = name_line[2]

			#Close file
			names.close()
			print("Preparation of names-list done\n")

	#Open tab-file to write to
	output_taxa = open("../../data/data_model_preparation/taxonomy/eukaryote_species_taxonomy.tab", "w")

	#Initialize
	species_IDs = []
	count_euks = 0

	#Get list of node IDs for all species (not other ranks)
	for node in nodes_dict:
		if nodes_dict[node][1] == "species":
			species_IDs.append(node)

	print("Number of species in NCBI Taxonomy Database: ", len(species_IDs))

	#Loop over each species ID
	for species_ID in species_IDs:

		#Initialize for each species
		searching = True
		tax_IDs_list = []

		#Rename species ID
		node_ID = species_ID

		while searching:
			#Append node ID of current node
			tax_IDs_list.append(node_ID)

			#Get parent node ID of current node, overwrite
			node_ID = nodes_dict[node_ID][0]

			#Node ID numbers correspond to reaching the highest level 
			#(above eukarya, bacteria etc. - Taxonomy for a given species finished here)
			if node_ID in ["131567", "1"]:
				searching = False
			
			#If species in the eukaryotic domain (node ID 2759 is the eukaryotic domain)
			#NetStartPro: If needing the same kind for bacteria and/or archaea, replace with tax ID for these domains!!
			elif node_ID == "2759":
				#Append node ID 2759 to list
				tax_IDs_list.append(node_ID)

				#Eukaryotic species identified
				count_euks += 1

				#Write species' scientific name to outfile
				species = tax_IDs_list[0]
				output_taxa.write(names_dict[species] + "\t|")
			
				#Write the entire taxonomic distribution of species to outfile
				for node_id in tax_IDs_list:
					output_taxa.write("\t" + nodes_dict[node_id][1] + "\t" + node_id + "\t" + names_dict[node_id] + "\t|")
				output_taxa.write("\n")

				#Stop loop
				searching = False

				#Write status massage
				if count_euks % 20000 == 0:
					print("Processed", count_euks, "eukaryotic entries.", sep = " ")

	#Close outfile
	output_taxa.close()

sort_taxonomy_eukaryota()