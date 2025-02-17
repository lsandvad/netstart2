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
## Input Organism Names

| Species/Phylum                  | Input Argument          |
|---------------------------------|-------------------------|
| Alligator mississippiensis      | a_mississippiensis      |
| Anolis carolinensis             | a_carolinensis          |
| Anopheles gambiae               | a_gambiae               |
| Apis mellifera                  | a_mellifera             |
| Arabidopsis thaliana            | a_thaliana              |
| Aspergillus nidulans            | a_nidulans              |
| Bos taurus                      | b_taurus                |
| Brachypodium distachyon         | b_distachyon            |
| Caenorhabditis elegans          | c_elegans               |
| Canis lupus                     | c_lupus                 |
| Columba livia                   | c_livia                 |
| Coprinopsis cinerea             | c_cinerea               |
| Cryptococcus neoformans         | c_neoformans            |
| Danio rerio                     | d_rerio                 |
| Daphnia carinata                | d_carinata              |
| Dictyostelium discoideum        | d_discoideum            |
| Drosophila melanogaster         | d_melanogaster          |
| Eimeria maxima                  | e_maxima                |
| Entamoeba histolytica           | e_histolytica           |
| Equus caballus                  | e_caballus              |
| Gallus gallus                   | g_gallus                |
| Giardia intestinalis            | g_intestinalis          |
| Glycine max                     | g_max                   |
| Gorilla gorilla                 | g_gorilla               |
| Homo sapiens                    | h_sapiens               |
| Hordeum vulgare                 | h_vulgare               |
| Leishmania donovani             | l_donovani              |
| Lotus japonicus                 | l_japonicus             |
| Manduca sexta                   | m_sexta                 |
| Medicago truncatula             | m_truncatula            |
| Mus musculus                    | m_musculus              |
| Neurospora crassa               | n_crassa                |
| Nicotiana tabacum               | n_tabacum               |
| Oreochromis niloticus           | o_niloticus             |
| Oryctolagus cuniculus           | o_cuniculus             |
| Oryza sativa                    | o_sativa                |
| Oryzias latipes                 | o_latipes               |
| Ovis aries                      | o_aries                 |
| Pan troglodytes                 | p_troglodytes           |
| Phoenix dactylifera             | p_dactylifera           |
| Plasmodium falciparum           | p_falciparum            |
| Rattus norvegicus               | r_norvegicus            |
| Rhizophagus irregularis         | r_irregularis           |
| Saccharomyces cerevisiae        | s_cerevisiae            |
| Schizophyllum commune           | s_commune               |
| Schizosaccharomyces pombe       | s_pombe                 |
| Selaginella moellendorffii      | s_moellendorffii        |
| Setaria viridis                 | s_viridis               |
| Solanum lycopersicum            | s_lycopersicum          |
| Strongylocentrotus purpuratus   | s_purpuratus            |
| Sus scrofa                      | s_scrofa                |
| Taeniopygia guttata             | t_guttata               |
| Toxoplasma gondii               | t_gondii                |
| Tribolium castaneum             | t_castaneum             |
| Trichoplax adhaerens            | t_adhaerens             |
| Triticum aestivum               | t_aestivum              |
| Trypanosoma brucei              | t_brucei                |
| Ustilago maydis                 | u_maydis                |
| Xenopus laevis                  | x_laevis                |
| Zea mays                        | z_mays                  |
| Chordata                        | chordata                |
| Nematoda                        | nematoda                |
| Arthropoda                      | arthropoda              |
| Placozoa                        | placozoa                |
| Echinodermata                   | echinodermata           |
| Apicomplexa                     | apicomplexa             |
| Euglenozoa                      | euglenozoa              |
| Evosea                          | evosea                  |
| Fornicata                       | fornicata               |
| Streptophyta                    | streptophyta            |
| Ascomycota                      | ascomycota              |
| Basidiomycota                   | basidiomycota           |
| Mucoromycota                    | mucoromycota            |
| unknown                         | unknown                 |
