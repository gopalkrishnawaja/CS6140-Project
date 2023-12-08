# Malicious URL Classification Project

## Overview

This project focuses on the classification of URLs into three categories: Safe, Phishing, or Malware. The increasing threat of cyberattacks through malicious URLs necessitates advanced techniques beyond traditional methods. The project explores both manual feature extraction and automated feature extraction using machine learning and deep learning models.

## Project Structure

The project is organized into the following directories:

- **RawDatasets**: Contains the raw datasets used in **URL_Feature_Extraction.ipynb** notebook along with their sources.
  - benignDMOZ.csv: [DMOZ Open Directory Project](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OMV93V)
  - malwareOnline.csv: [Online Malicious URL Dataset](https://urlhaus.abuse.ch/browse/)
  - verified_phishing_online.csv: [Verified Phishing URLs Dataset](https://phishtank.org)

- **ExtractedFeaturesDataset**: Contains CSV files with manually extracted features for different URL classes.
  - Benign_Features_Final.csv
  - Malware_Features_Final.csv
  - Phish_Features_Final.csv

- **Notebooks**: Jupyter notebooks for different aspects of the project.
  - URL_Classification.ipynb: Notebook for training machine learning models and evaluating their performance.
  - URL_Feature_Extraction.ipynb: Notebook for extracting features from the raw datasets.

- **ModelWeights**: Directory containing pre-trained model weights in zip format.
  - about.txt: Information about the model weights.
  - model_embb.h5.zip: Zip file containing model weights.
  - model_pretrained_embb.h5.zip: Zip file containing pre-trained model weights.

- **Plots_n_Images**: Directory containing various plots related to feature analysis.
  - Box Plot for Domain_Shannon_Entropy.png
  - Box Plot for digitToLetter_ratio.png
  - Box_Plot for Path_Shannon_Entropy.png
  - Box_Plot for URL_Shannon_Entropy.png
  - Box_Plot for getLength.png
  - Box_Plot for numDigit.png
  - Box_Plot for numsubdomain.png
  - Feature importance for DecisionTree.png
  - Feature importance for RandomForest.png
  - Feature importance for XgBoost.png

- **ML_Project_Presentation.pdf**: Project Presentation.

- **ML_Project_Report.pdf**: The comprehensive report detailing the project's motivation, methodology, experimental setup, results, and conclusion.

- **project_tree.txt**: A text file providing the project directory tree structure.

- **requirements.txt**: File containing project dependencies for reproducibility.

- **README.md**: This file, providing an overview of the project, its structure, and instructions for running and reproducing the results.

## Running the Notebooks

1. Open the **URL_Feature_Extraction.ipynb** notebook to extract features from the raw datasets.
   - Ensure required CSV files are present in RawDatasets folder.
   - Run the cells in sequence to generate the CSV files with extracted features.
   - Ensure CSV files are generated in ExtractedFeaturesDataset folder.
     
2. Open the **URL_Classification.ipynb** notebook to train machine learning models.
   - Ensure required CSV files are present in ExtractedFeaturesDataset folder.
   - Follow the instructions in the notebook to load the extracted features and train the models.

## Results Summary

The results for best parameters for each of the models as the result of grid search using 10 fold cross validation are as follows :- 

| Model                | Best Parameters                    |
|----------------------|-----------------------------------|
| Logistic Regression  | C: 100                            |
| Decision Tree        | max depth: 20                      |
| Random Forest        | max depth: 20, n estimators: 200   |
| SVM                  | C: 10, kernel: rbf                 |
| XG Boost             | max depth: 5, n estimators: 100    |
| KNN                  | n neighbors: 3                    |

The performace for the models is as follows : -

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

Future enhancements could involve exploring imbalance reduction techniques like SMOTE and Near Miss Algorithm, testing the model on different datasets, and conducting A/B testing for model validation. We could also look into exploration of advanced deep learning architectures like attention mechanisms, transformers, or graph neural networks for enhanced performance. Transfer learning techniques could be evaluated to leverage knowledge across domains, and domain adaptation methods can be explored. Dynamic URL analysis over time and real-time processing optimizations are other avenues for exploration. Privacy-preserving methods, collaborative learning for knowledge sharing while preserving privac, Cross-language URL classification, ethical considerations, and addressing biases for fair and responsible classification are essential aspects to explore in future research.

## References

- [Towards Fighting Cybercrime: Malicious URL Attack Type Detection using Multiclass Classification](https://ieeexplore.ieee.org/document/9378029)
- [Phishing Website Detection using Machine Learning Algorithms](https://www.researchgate.net/publication/328541785_Phishing_Website_Detection_using_Machine_Learning_Algorithms)
- [URL Feature Engineering and Classification](https://medium.com/nerd-for-tech/url-feature-engineering-and-classification-66c0512fb34d)
