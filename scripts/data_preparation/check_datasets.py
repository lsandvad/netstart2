import gzip
import os

#Check that every sample has had an ATG being labelled correctly. 
filenames = os.listdir("../../data/data_model_preparation/datasets/non_TIS/mRNA")

for filename in filenames:
	with gzip.open(f"../../data/data_model_preparation/datasets/non_TIS/mRNA/"+filename, "rt") as file:
		file.readline()
		#Iterate line by line
		for line in file:
			line_sep = line.split(",")
			if line_sep[0][int(line_sep[1]):int(line_sep[1])+3] != "ATG":
				print("Error!")

print("Completed check for non-TIS mRNA data")

print("Runnnig check")
filenames = os.listdir("../../data/data_model_preparation/datasets/TIS")

for filename in filenames:
	print(filename)
	with gzip.open(f"../../data/data_model_preparation/datasets/TIS/"+filename, "rt") as file:
		file.readline()
		#Iterate line by line
		for line in file:
			line_sep = line.split(",")
			if line_sep[0][int(line_sep[1]):int(line_sep[1])+3] != "ATG":
				print("Error!")

print("Completed check for TIS mRNA data")


filenames = os.listdir("../../data/data_model_preparation/datasets/non_TIS/introns")

for filename in filenames:
	with gzip.open(f"../../data/data_model_preparation/datasets/non_TIS/introns/"+filename, "rt") as file:
		file.readline()
		#Iterate line by line
		for line in file:
			line_sep = line.split(",")
			if line_sep[0][int(line_sep[1]):int(line_sep[1])+3] != "ATG":
				print("Error!")

print("Completed check for non-TIS intron data")

filenames = os.listdir("../../data/data_model_preparation/datasets/non_TIS/intergenic")

for filename in filenames:
	with gzip.open(f"../../data/data_model_preparation/datasets/non_TIS/intergenic/"+filename, "rt") as file:
		file.readline()
		#Iterate line by line
		for line in file:
			line_sep = line.split(",")
			if line_sep[0][int(line_sep[1]):int(line_sep[1])+3] != "ATG":
				print("Error!")

print("Completed check for non-TIS intergenic data")
