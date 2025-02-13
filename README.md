# NetStart 2.0 Development Code
NetStart 2.0 is a deep learning-based model that predicts canonical translation initiation sites in mRNA transcripts in species across the eukaryotic domain.

## About
NetStart 2.0 integrates the ESM-2 protein language model for assessing transitions from non-coding to coding regions with local sequence context and taxonomical information. 


## NetStart 2.0 Online Server
For smaller datasets, the NetStart 2.0 prediction server is available for use [here](https://services.healthtech.dtu.dk/services/NetStart-2.0/). 

## Usage Instructions
NetStart 2.0 can be run locally by cloning this repository and installing the required packages. From the project root, NetStart 2.0 can be run using: \
```
python3 ./predict_with_netstart2.py -o ORIGIN -in INPUT_FILENAME 
```

Origin of sequence (-o) and path and name of input file (-in) are required for running NetStart 2.0.

### Requirements
NetStart 2.0 has been developed with python and the following package versions: \
* torch==1.12.1
* numpy==1.23.5
* pandas==2.0.3
* tqdm==4.62.3
* transformers==4.36.0

