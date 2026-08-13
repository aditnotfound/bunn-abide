# ABIDE graph-model protocol comparison

This table records only values verified from the cited paper or its official
record. It is not a meta-analysis. The studies do not share one estimand.

| Study | Cohort and representation | Evaluation | Reported result | Why it is not a direct score comparison |
| --- | --- | --- | --- | --- |
| This study | 754 ABIDE-I participants; C-PAC no-GSR; AAL-116; Fisher-z connectivity rows; participant brain graphs | nested leave-one-site-out; equal-site inference; test sites excluded from tuning and stopping | elastic net BA 0.6401; best descriptive GCN density BA 0.6166; GCN curve BA 0.5945; BuNN curve BA 0.5849 | primary metric is equal-site balanced accuracy and the central estimand is a density curve |
| Wang, Li, and Hu (2021) | 1,057 participants; CCS; CC200 raw ROI time series; training-derived k-NN ROI graph; five graph layers and temporal averaging | paper reports leave-one-site-out ordinary accuracy | 71.6% mean accuracy at k=5 | different atlas, preprocessing, time-series input, architecture, cohort, and metric; released code uses the held-out site for epoch/checkpoint selection |
| Parisot et al. (2018) | 871 participants; connectivity features on a population graph whose nodes are participants and whose edges use sex/site information | 10-fold transductive cross-validation | 70.4% accuracy and 0.75 AUC | participant-level population graph, phenotypic edges, and mixed-site folds do not test unseen-site brain-graph generalization |
| Yang et al. (2022) | 1,035 ABIDE-I participants; functional graphs plus graphlet counts | whole-dataset and site-specific GCN evaluation as reported by the paper | 64.27% average accuracy; 75.9% highest site-specific accuracy | graphlet features, cohort, averaging, and ordinary accuracy differ; the highest single-site value is not an overall estimate |
| Han et al. (2026) | 1,009 participants; C-PAC; CC200 connectivity matrices; several graph, non-graph, and classical models | controlled benchmark, ABIDE reported using AUROC; density ablations average ten runs | qualitative result: simple models often matched or exceeded graph models and aggregation degraded with density | different atlas, cohort, metric, split implementation, and model family; useful as a mechanistic comparison rather than a numerical target |

## Sources

- Wang, L., Li, K., and Hu, X. P. (2021). *Graph convolutional network for fMRI analysis based on connectivity neighborhood*. Network Neuroscience, 5(1), 83-95. https://doi.org/10.1162/netn_a_00171
- Parisot, S. et al. (2018). *Disease prediction using graph convolutional networks: Application to autism spectrum disorder and Alzheimer's disease*. Medical Image Analysis, 48, 117-130. https://doi.org/10.1016/j.media.2018.06.001
- Yang, T. et al. (2022). *Classification of Autism Spectrum Disorder Using rs-fMRI Data and Graph Convolutional Networks*. IEEE Big Data 2022, 3131-3138. https://doi.org/10.1109/BigData55660.2022.10021070
- Han, K. et al. (2026). *Rethinking functional brain connectome analysis: do graph deep learning models help?* npj Artificial Intelligence, 2, 19. https://doi.org/10.1038/s44387-025-00067-x

