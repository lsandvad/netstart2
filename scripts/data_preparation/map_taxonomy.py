#!/usr/bin/env python3

import pandas as pd
import json

#######################################################################
###Get species' taxonomical information in correct formats for model###
#######################################################################

#Get list of species in dataset
df_dataset_species = pd.read_csv("../../data/data_raw/species_information/species_groups.csv")
species_list = list(df_dataset_species["Species"])
species_list = ['canis_lupus' if species == 'canis_lupus_familiaris' else species for species in species_list]
formatted_species_list = [name.replace('_', ' ').capitalize() for name in species_list]

#Open generated tax file
infile_tax = open("../../data/data_model_preparation/taxonomy/eukaryote_species_taxonomy.tab", "r")

#Initialize
dict_species_names = dict()
dict_kingdom = dict()
dict_phylum = dict()
dict_class = dict()
dict_order = dict()
dict_family = dict()
dict_genus = dict()

for line in infile_tax:

	tax_levels_data = line.split("|")
	species_info = tax_levels_data[1]

	species_name = species_info.split("\t")[3]
	species_tax_id = species_info.split("\t")[2]

	if species_name in formatted_species_list:

		#Get mapping from species ID to species name
		dict_species_names[species_name] = species_tax_id

		#Get mappings from species to its respective taxonomical ranks
		kingdom_line = next((line for line in tax_levels_data if line.startswith('\tkingdom\t')), None)
		kingdom_name = kingdom_line.strip().split('\t')[-2] if kingdom_line else '0'
		dict_kingdom[species_tax_id] = kingdom_name

		phylum_line = next((line for line in tax_levels_data if line.startswith('\tphylum\t')), None)
		phylum_name = phylum_line.strip().split('\t')[-2] if phylum_line else '0'
		dict_phylum[species_tax_id] = phylum_name

		class_line = next((line for line in tax_levels_data if line.startswith('\tclass\t')), None)
		class_name = class_line.strip().split('\t')[-2] if class_line else '0'
		dict_class[species_tax_id] = class_name

		order_line = next((line for line in tax_levels_data if line.startswith('\torder\t')), None)
		order_name = order_line.strip().split('\t')[-2] if order_line else '0'
		dict_order[species_tax_id] = order_name

		family_line = next((line for line in tax_levels_data if line.startswith('\tfamily\t')), None)
		family_name = family_line.strip().split('\t')[-2] if family_line else '0'
		dict_family[species_tax_id] = family_name

		genus_line = next((line for line in tax_levels_data if line.startswith('\tgenus\t')), None)
		genus_name = genus_line.strip().split('\t')[-2] if genus_line else '0'
		dict_genus[species_tax_id] = genus_name

#Write taxonomical mappings to files
with open("../../data/data_model/taxonomy/species_names.json", 'w') as json_file:
    json.dump(dict_species_names, json_file, indent = 4) 

with open("../../data/data_model/taxonomy/species_to_kingdom.json", 'w') as json_file:
    json.dump(dict_kingdom, json_file, indent = 4) 

with open("../../data/data_model/taxonomy/species_to_phylum.json", 'w') as json_file:
    json.dump(dict_phylum, json_file, indent = 4) 

with open("../../data/data_model/taxonomy/species_to_class.json", 'w') as json_file:
    json.dump(dict_class, json_file, indent = 4) 

with open("../../data/data_model/taxonomy/species_to_order.json", 'w') as json_file:
    json.dump(dict_order, json_file, indent = 4) 

with open("../../data/data_model/taxonomy/species_to_family.json", 'w') as json_file:
    json.dump(dict_family, json_file, indent = 4) 

with open("../../data/data_model/taxonomy/species_to_genus.json", 'w') as json_file:
    json.dump(dict_genus, json_file, indent = 4) 

infile_tax.close()