# NetStart 2.0: Prediction of Eukaryotic Translation Initiation Sites Using Protein Language Modeling
NetStart 2.0 is a deep learning-based model that predicts canonical translation initiation sites in mRNA transcripts in species across the eukaryotic domain.

## About
NetStart 2.0 integrates the ESM-2 protein language model for assessing transitions from non-coding to coding regions with local sequence context and taxonomical information. 

## NetStart 2.0 Online Server
For smaller datasets, the NetStart 2.0 prediction server is available for use [here](https://services.healthtech.dtu.dk/services/NetStart-2.0/). 

### Requirements
NetStart 2.0 has been developed in Python, with the following package versions: \
* torch==1.12.1
* numpy==1.23.5
* pandas==2.0.3
* tqdm==4.62.3
* transformers==4.36.0 

## Usage Instructions
NetStart 2.0 can be run locally via the command line by cloning this repository and installing the required packages. From the project root, NetStart 2.0 can be run using: 
```
python3 ./predict_with_netstart2.py [optional arguments] -o ORIGIN -in INPUT_FILENAME 
```
The origin of the sequence and the input file (fasta format) are required. Besides these, NetStart 2.0 can take a range of optional arguments: 

| Input Argument                  | Description                                                                                                                                          |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-out`, `--output_filename`     | Output file name without file extension.                                                                                                             |
| `--output_results`              | Output format of predictions. The predictions can be provided in three versions: `all` returns the Predicted probabilities for all ATGs in the input sequence(s). `max_prob` returns only the ATG with the highest predicted probability for being a translation initiation site for each input sequence. `threshold` returns all ATGs having a predicted probability of being a translation initiation site above the specified threshold. Default: `all`.|            
| `--threshold`                | Set the threshold for filtering predictions. This argument only works with `--output_results threshold`. Default: `0.625`.            |
| `--gzip_outfile`             | Specify if output file should be gzipped. Default: `False`.            |
| `--batch_size`               | The batch size for running predictions. Default: `64`.            |
| `-o`, `--origin`             | The sequence origin (See table XXX for options).            |
| `-in`, `--input_filename`    | The input file in FASTA format. The input file can also be in gzipped format with .gz-extension            |


 
 The predictions are returned as a .csv file. 



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
