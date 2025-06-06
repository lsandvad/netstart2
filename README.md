# NetStart 2.0: Prediction of Eukaryotic Translation Initiation Sites Using a Protein Language Model
NetStart 2.0 is a deep learning-based model that predicts canonical translation initiation sites in mRNA transcripts in species across the eukaryotic domain.

## About
NetStart 2.0 integrates the ESM-2 protein language model for assessing transitions from non-coding to coding regions with local sequence context and taxonomical information. 

## NetStart 2.0 Online Server
For smaller datasets, the NetStart 2.0 prediction server is available for use [here](https://services.healthtech.dtu.dk/services/NetStart-2.0/). 

### Training and test sets
The datasets used for training and testing NetStart 2.0 can be downloaded from the [prediction server](https://services.healthtech.dtu.dk/services/NetStart-2.0/) from the section **Data**.

## Run NetStart 2.0 Locally
### Requirements
NetStart 2.0 has been developed in Python, with the following package versions:
* torch==1.12.1
* numpy==1.23.5
* pandas==2.0.3
* tqdm==4.62.3
* transformers==4.36.0 

### Usage Instructions
NetStart 2.0 can be run via the command line by cloning this repository and installing the required packages. 
To test the installation, run the following command from project root:
```
python3 ./predict_with_netstart2.py -o chordata -in ./data_example/input_file.fasta
```

NetStart 2.0 can be run to predict on your own data using the general command:
```
python3 ./predict_with_netstart2.py [optional arguments] -o ORIGIN -in INPUT_FILENAME 
```
Please note that the program uses the information in the /src directory. 

The sequence origin and input file (in FASTA format) are required. Additionally, NetStart 2.0 accepts a range of optional arguments:

| Input Argument                      | Description                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-out`, `--output_filename`     | Name of the output file without the file extension.                                                                                                   |
| `--output_results`              | Format of the prediction results. Options are: `all` (returns predicted probabilities for all ATGs in the input sequence(s)), `max_prob` (returns only the ATG with the highest predicted probability for each input sequence), and `threshold` (returns all ATGs with a predicted probability above the specified threshold). Default: `all`. |
| `--threshold`                   | Sets the threshold for filtering predictions. This option is only applicable when `--output_results` is set to `threshold`. Default value: `0.625`.   |
| `--gzip_outfile`                | Specifies whether the output file should be gzipped. Default value: `False`.                                                                          |
| `--batch_size`                  | Specifies the number of samples to process together in a single pass during prediction. Default value: `64`.                                     |
| `-o`, `--origin`                | Origin of the sequence(s) (refer to the [Input Sequence Origin Specifications](#input-sequence-origin-specifications) table for options).             |
| `-in`, `--input_filename`       | Input file in FASTA format. The allowed input alphabet is A, C, G, T, U and N (unknown). All the other letters will be converted to N before processing. T and U are treated as equivalent. The input file can also be provided in gzipped version with a .gz extension.                                                                        |
|`--compute_device` | Which hardware accelerator to use. Options are:  `cuda` (NVIDIA GPU), `mps` (Apple Silicon), or `cpu`. The program will automatically fall back to CPU if the requested device is unavailable. |
|`--include_reverse_complement` | Specifies whether NetStart 2.0 should predict on both strands. We recommend using this option, if you want to use NetStart 2.0 to predict on genomic sequences. Default: `False`. |

The predictions are returned as a .csv file, see [Output format description](#output-format-description) for description. 



### Input Sequence Origin Specifications
| **Sequence Origin**             | **Input Argument** (`-o`) |
|---------------------------------|---------------------------|
| **Chordata**                    | `chordata`                |
| *Alligator mississippiensis*    | `a_mississippiensis`      |
| *Anolis carolinensis*           | `a_carolinensis`          |
| *Bos taurus*                    | `b_taurus`                |
| *Canis lupus*                   | `c_lupus`                 |
| *Columba livia*                 | `c_livia`                 |
| *Danio rerio*                   | `d_rerio`                 |
| *Equus caballus*                | `e_caballus`              |
| *Gallus gallus*                 | `g_gallus`                |
| *Gorilla gorilla*               | `g_gorilla`               |
| *Homo sapiens*                  | `h_sapiens`               |
| *Mus musculus*                  | `m_musculus`              |
| *Oryctolagus cuniculus*         | `o_cuniculus`             |
| *Oreochromis niloticus*         | `o_niloticus`             |
| *Oryzias latipes*               | `o_latipes`               |
| *Ovis aries*                    | `o_aries`                 |
| *Pan troglodytes*               | `p_troglodytes`           |
| *Rattus norvegicus*             | `r_norvegicus`            |
| *Sus scrofa*                    | `s_scrofa`                |
| *Taeniopygia guttata*           | `t_guttata`               |
| *Xenopus laevis*                | `x_laevis`                |
| **Arthropoda**                  | `arthropoda`              |
| *Apis mellifera*                | `a_mellifera`             |
| *Anopheles gambiae*             | `a_gambiae`               |
| *Daphnia carinata*              | `d_carinata`              |
| *Drosophila melanogaster*       | `d_melanogaster`          |
| *Manduca sexta*                 | `m_sexta`                 |
| *Tribolium castaneum*           | `t_castaneum`             |
| **Nematoda**                    | `nematoda`                |
| *Caenorhabditis elegans*        | `c_elegans`               |
| **Placozoa**                    | `placozoa`                |
| *Trichoplax adhaerens*          | `t_adhaerens`             |
| **Echinodermata**               | `echinodermata`           |
| *Strongylocentrotus purpuratus* | `s_purpuratus`            |
| **Apicomplexa**                 | `apicomplexa`             |
| *Plasmodium falciparum*         | `p_falciparum`            |
| *Toxoplasma gondii*             | `t_gondii`                |
| *Eimeria maxima*                | `e_maxima`                |
| **Euglenozoa**                  | `euglenozoa`              |
| *Trypanosoma brucei*            | `t_brucei`                |
| *Leishmania donovani*           | `l_donovani`              |
| **Evosea**                      | `evosea`                  |
| *Dictyostelium discoideum*      | `d_discoideum`            |
| *Entamoeba histolytica*         | `e_histolytica`           |
| **Fornicata**                   | `fornicata`               |
| *Giardia intestinalis*          | `g_intestinalis`          |
| **Streptophyta**                | `streptophyta`            |
| *Arabidopsis thaliana*          | `a_thaliana`              |
| *Brachypodium distachyon*       | `b_distachyon`            |
| *Glycine max*                   | `g_max`                   |
| *Hordeum vulgare*               | `h_vulgare`               |
| *Lotus japonicus*               | `l_japonicus`             |
| *Medicago truncatula*           | `m_truncatula`            |
| *Nicotiana tabacum*             | `n_tabacum`               |
| *Oryza sativa*                  | `o_sativa`                |
| *Phoenix dactylifera*           | `p_dactylifera`           |
| *Selaginella moellendorffii*    | `s_moellendorffii`        |
| *Setaria viridis*               | `s_viridis`               |
| *Solanum lycopersicum*          | `s_lycopersicum`          |
| *Triticum aestivum*             | `t_aestivum`              |
| *Zea mays*                      | `z_mays`                  |
| **Ascomycota**                  | `ascomycota`              |
| *Aspergillus nidulans*          | `a_nidulans`              |
| *Neurospora crassa*             | `n_crassa`                |
| *Saccharomyces cerevisiae*      | `s_cerevisiae`            |
| *Schizosaccharomyces pombe*     | `s_pombe`                 |
| **Basidiomycota**               | `basidiomycota`           |
| *Coprinopsis cinerea*           | `c_cinerea`               |
| *Cryptococcus neoformans*       | `c_neoformans`            |
| *Schizophyllum commune*         | `s_commune`               |
| *Ustilago maydis*               | `u_maydis`                |
| **Mucoromycota**                | `mucoromycota`            |
| *Rhizophagus irregularis*       | `r_irregularis`           |
| **Unknown**                     | `unknown`                 |



## Output Format Description
The output is provided as a .csv file which contains the following information attributes for each prediction (ATG):
| **Attribute**           | **Description**                          |
|-------------------------|------------------------------------------|
| `origin`                | Specifies the origin of the sequence predicted on (provided by the user). |
| `atg_pos`               | States the position of the ATG predicted upon (corresponds to the A in the codon). |
| `entry_line`            | Specifies the fasta header line of the specific sequence. |
| `preds`                 | Provides the predicted probability of the specific ATG being a translation initiation site (in the range [0.0, 1.0]).
| `stop_codon_position`   | States the position of the first in-frame stop codon relative to the ATG prediction upon (position corresponds to the first position of the stop codon). |
| `peptide_len`           | States the length of the hypothetical peptide. |
| `strand`                | States the strand predicted upon (+ denotes the template strand, - denotes the complement strand). | 
