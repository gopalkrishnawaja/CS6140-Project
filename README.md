# Malicious URL Classification Project

## Overview

This project focuses on the classification of URLs into three categories: Safe, Phishing, or Malware. The increasing threat of cyberattacks through malicious URLs necessitates advanced techniques beyond traditional methods. The project explores both manual feature extraction and automated feature extraction using machine learning and deep learning models.

## Project Structure

The project is organized into the following directories:

- **RawDatasets**: Contains the raw datasets used in the project.
  - benignDMOZ.csv
  - malwareOnline.csv
  - verified_phishing_online.csv
    
- **ExtractedFeaturesDataset**: Contains CSV files with manually extracted features for benign, malware, and phishing URLs.
  - Benign_Features_Final.csv
  - Malware_Features_Final.csv
  - Phish_Features_Final.csv

- **Notebooks**: Jupyter notebooks for different aspects of the project.
  - **URL_Classification.ipynb**: Notebook for training machine learning models and evaluating their performance.
  - **URL_Feature_Extraction.ipynb**: Notebook for extracting features from the raw datasets.

- **ML_Project_Report.pdf**: The comprehensive report detailing the project's motivation, methodology, experimental setup, results, and conclusion.

- **README.md**: This file, providing an overview of the project, its structure, and instructions for running and reproducing the results.

## Running the Notebooks

1. Open the **URL_Feature_Extraction.ipynb** notebook to extract features from the raw datasets.
   - Run the cells in sequence to generate the CSV files with extracted features.
3. Open the **URL_Classification.ipynb** notebook to train machine learning models.
   - Follow the instructions in the notebook to load the extracted features and train the models.

## Results Summary

| Index | Model                         | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|-------|------------------------------|----------------|---------------|-----------|--------|----------|
| 0     | Logistic Regression           | 0.8823         | 0.8754        | 0.8760    | 0.8754 | 0.8756   |
| 1     | Decision Tree                 | 0.9857         | 0.9371        | 0.9373    | 0.9371 | 0.9371   |
| 2     | Random Forest                 | 0.9878         | 0.9550        | 0.9552    | 0.9550 | 0.9550   |
| 3     | SVM                           | 0.9116         | 0.9046        | 0.9081    | 0.9046 | 0.9051   |
| 4     | XG Boost                      | 0.9701         | 0.9546        | 0.9548    | 0.9546 | 0.9547   |
| 5     | KNN                           | 0.9464         | 0.9074        | 0.9084    | 0.9074 | 0.9076   |
| 6     | Pretrained Embeddings + NN    | 0.9922         | 0.9778        | 0.9779    | 0.9778 | **0.9778**   |
| 7     | Normal Embeddings + NN        | 0.5561         | 0.5511        | 0.3832    | 0.5511 | 0.4061   |

## Future Work

Future enhancements could involve exploring imbalance reduction techniques like SMOTE and Near Miss Algorithm, testing the model on different datasets, and conducting A/B testing for model validation.

## References

- [Towards Fighting Cybercrime: Malicious URL Attack Type Detection using Multiclass Classification](https://ieeexplore.ieee.org/document/9378029)
- [Phishing Website Detection using Machine Learning Algorithms](https://www.researchgate.net/publication/328541785_Phishing_Website_Detection_using_Machine_Learning_Algorithms)
- [URL Feature Engineering and Classification](https://medium.com/nerd-for-tech/url-feature-engineering-and-classification-66c0512fb34d)
