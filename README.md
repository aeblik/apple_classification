# apple_classification

The file FE_pipeline is used for automatic feature extraction, data augmentation and balancing. It also generates a boxplot for each extracted feature.

The file utils_data contains functions for data balancing and augmentation.

The file utils_FE contains functions for the feature extraction and generation of boxplots of each feature.

The jupyter notebooks SVM and RF were used to train and evaluate the model - once with color features and once without. 

Before running the code, make sure your directory structure is fitting with the process in "process_pcds_and_extract_features" from the module utlils_FE. 

visualize.py can be used to visualize chosen features using open3d. Again the directory structury might need to be adapted. This file will visualize every single pcd in every subdirectory according to the directory structure I used, one by one.
