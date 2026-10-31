# Crime Type Prediction from Spatiotemporal Police Data

MSc team project (2 members), INM701 Introduction to Artificial Intelligence, City St George's, University of London (Dec 2025). This repository holds Badrinarayan Nandakumar's notebooks.

A multiclass crime-type prediction pipeline on 18.1 million UK police.uk records covering 36 months (Sep 2022 – Aug 2025).

- **Data engineering:** merging and cleaning 36 monthly files. An HDF5 chunked-storage cache shrank the 7 GB dataset by 91% and cut re-training time for new configurations by up to 65%.
- **Models:** KNN, SVM (with hyperparameter optimisation), Random Forest, 1D-CNN and MLP. The tuned MLP reached about 64% accuracy, nearly double the 34.5% majority-class share, using class weighting, early stopping, learning-rate scheduling and tf.data pipelines.
- **Evaluation:** macro and weighted F1, and ROC-AUC.

Notebooks are numbered in pipeline order (`01-*` data, `02-*` models). The data is not included; download it from [data.police.uk](https://data.police.uk/data/), which is available under the Open Government Licence.

**Skills:** Python · Pandas · scikit-learn · TensorFlow/Keras · HDF5 (h5py) · tf.data
