# Identification of clinical disease trajectories in neurodegenerative disorders with natural language processing

https://www.medrxiv.org/content/10.1101/2022.09.22.22280158v1

---

## Introduction
This repository contains all the code associated with the publication: "Identification of clinical disease trajectories in neurodegenerative disorders with natural language processing". The aim of this project was to use large language models to standardize the donor clinical summaries written by the Netherlands Brain Bank (NBB). 

Files included here are subdivided into multiple folders:  
- GruD: Contains code for implementing, running, and analysing the GruD model from [this](https://www.nature.com/articles/s41598-018-24271-9) publication.  
- Dimensionality reduction: All scripts related to performing the Seurat dimensionality reduction analysis and PCA.
- ML: Scripts related to ML models such as BOW, SVC, transformer models. Also, scripts for the analysis of the result of the models and the processing of the predictions.
- Plots: Scripts used for some of the visualizations used in the paper such as the scattermap and heatmap plots.
- other: Other important scripts for task such as: calculating interoperator agreement, some helper functions, and donor preprocessing.

## Data availability
Data is made available via Zenodo. Data that can be found here includes the donor general information, clinical disease trajectories, NLP training data, and neuropathological diagnosis metadata. Data can be found on the following link: [10.5281/zenodo.10526891](https://zenodo.org/records/10526891)


## Model availability
The trained large language model for the Clinical History and the GruD model are publicly available on Huggingface via the following links.

| Model | Description | Link |
|----------|----------|----------|
| Mekkes Clinical History | Large language model based on PubMedBert | [clinical history](https://huggingface.co/NND-project/Clinical_History_Mekkes_PubmedBert) |
| Gru-D prediction model | Gru-D prediction model to predict neuropathological diagnosis | [GruD model](https://huggingface.co/NND-project/Clinical_History_Mekkes_GruD) |

## Requirements
| Library | Version | 
|----------|----------|
| Pandas |  1.5.3 | 
| Simpletransformers | 0.64.5 | 
| More | ??? | 

## Citation
<needs to be fully updated>
  
```
@article{mekkesetal,
    title = {Identification of clinical disease trajectories in neurodegenerative disorders with natural language processing},
    author = {Nienke Mekkes and Minke Groot and Eric Hoekstra and Alyse de Boer and
              Sander Bouwman and Sophie Wehrens and Megan K. Herbert and Dennis Wever and
              Annemieke Rozemuller and Bart J.L. Eggen and Inge Huitinga, Inge R. Holtman},
    journal={Nature Medicine TBA},
    year = {2024},
}
```