# Apple Classification using 3D Point Cloud Data

This repository contains code and utilities for the automated classification of apple varieties using LiDAR-captured 3D point cloud data. The primary goal is to extract meaningful geometric and morphological features and use them for training and evaluating machine learning models. 

## Repository Structure

### Feature Extraction Pipeline
- **`FE_pipeline.py`**  
  Automates the processes of feature extraction, data augmentation, and balancing. The according functions are implemented from the utility modules.

### Utility Modules
- **`utils_data.py`**  
  Contains helper functions for data augmentation and balancing.

- **`utils_FE.py`**  
  Provides functions for feature extraction and boxplot generation.

### Machine Learning Models
- **`SVM.ipynb`**  
  A Jupyter Notebook to train and evaluate the Support Vector Machine (SVM) model. Includes experiments both with and without color features.

- **`RF.ipynb`**  
  A Jupyter Notebook to train and evaluate the Random Forest (RF) model. Similar to the SVM notebook, it includes tests with and without color features.

### Visualization
- **`visualize.py`**  
  Uses Open3D to visualize selected features. The script iterates through all point cloud files in the directory structure, displaying one by one. Note: The directory structure may need adjustment to fit your specific dataset.

### Directory Structure
Ensure your directory structure aligns with the expected process outlined in the `process_pcds_and_extract_features` function in the **`utils_FE`** module. This is critical for proper feature extraction and visualization.

### Dependencies
All required Python libraries are listed in the `requirements.txt` file. Please install them using:

```bash
pip install -r requirements.txt
