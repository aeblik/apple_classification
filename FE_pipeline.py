import os
import open3d as o3d
from utils_FE import process_pcds_and_extract_features, generate_boxplots
from utils_data import augment_features, balance_data_with_borderlinesmote
import pandas as pd

if __name__ == "__main__":
    base_folder_path = "C:/Daten/PA2/Code/base23_set1/2023_filtered"  # Directory containing the PCD files
    output_file = "C:/Daten/PA2/Code/Output/apple_features_FE_Final_REVISED_1012.csv"  # Path to save the feature CSV
    feature_plots_folder = "C:/Daten/PA2/Code/Output/feature_plots_FE_Final_REVISED_1012"  # Folder to save feature plots
    variety_mapping_file = "C:/Daten/PA2/Code/base23_set1/base23_set1_names.csv"  # CSV with variety mapping¨
    augmented_output_file = "C:/Daten/PA2/Code/Output/apple_features_augmented_FE_Final_REVISED_1012.csv" # Path to save the augmented feature
    balanced_output_file =  "C:/Daten/PA2/Code/Output/apple_features_balanced_FE_Final_REVISED_1012.csv"  # Path to save the balanced dataset 

    # Ensure output directories exist
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    if not os.path.exists(feature_plots_folder):
        os.makedirs(feature_plots_folder)
    

    # Process all PCDs and extract features
    print("Starting feature extraction...")
    # process_pcds_and_extract_features(base_folder_path, variety_mapping_file, output_file)
    # print(f"Feature extraction completed. Features saved to {output_file}.")
    # generate_boxplots(output_file, feature_plots_folder)
    print(f"Box plots generated and saved to {feature_plots_folder}.")
    df = pd.read_csv(output_file)
    print("Augmenting features...")
    df_augmented = augment_features(df, augmentation_factor=3, target_column="Variety", categorical_columns=None, augmentation_methods=["noise", "scaling"])
    df_augmented.to_csv(augmented_output_file, index=False)
    print(f"Augmented features saved to {augmented_output_file}.") 

    # Separate features, target, and metadata
    non_numerical_columns = ["Tree", "Apple"]  # Columns to exclude from balancing
    numerical_features = df_augmented.drop(columns=["Variety"] + non_numerical_columns)
    target = df_augmented["Variety"]
    metadata = df_augmented[non_numerical_columns]

    # Balance the dataset with BorderlineSMOTE
    print("Balancing the dataset with BorderlineSMOTE...")
    X_balanced, y_balanced = balance_data_with_borderlinesmote(numerical_features, target)

    # Add metadata back to the balanced dataset
    metadata_balanced = pd.concat([metadata] * (len(X_balanced) // len(metadata)), ignore_index=True)[:len(X_balanced)]
    df_balanced = pd.concat([metadata_balanced, X_balanced, y_balanced], axis=1)

    # Save the balanced dataset
    df_balanced.to_csv(balanced_output_file, index=False)
    print(f"Balanced dataset saved to {balanced_output_file}.")

