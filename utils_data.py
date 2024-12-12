from imblearn.over_sampling import BorderlineSMOTE
import pandas as pd
import numpy as np

def balance_data_with_borderlinesmote(X, y):
    """
    Balances the dataset using BorderlineSMOTE.

    Parameters:
    - X (pd.DataFrame): Feature data.
    - y (pd.Series): Target labels.

    Returns:
    - X_balanced (pd.DataFrame): Balanced feature data.
    - y_balanced (pd.Series): Balanced target labels.
    """
    smote = BorderlineSMOTE(random_state=42, k_neighbors=4)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced, name=y.name)

def augment_features(
    df, 
    augmentation_factor, 
    target_column=None, 
    categorical_columns=None, 
    augmentation_methods=["noise", "scaling", "categorical_augmentation"]
):
    """
    Augments the extracted features to introduce variability and balance classes,
    while handling explicitly defined categorical features correctly.

    Parameters:
    - df (pd.DataFrame): Feature dataset.
    - augmentation_factor (int): Number of synthetic samples to generate per original sample.
    - target_column (str): Column name of the target variable (optional, e.g., for preserving labels).
    - categorical_columns (list): List of columns to treat as categorical.
    - augmentation_methods (list): List of augmentation methods to apply (e.g., "noise", "scaling", "categorical_augmentation").

    Returns:
    - augmented_df (pd.DataFrame): Augmented dataset with new synthetic samples.
    """
    # Separate numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude target column from augmentation
    if target_column:
        numeric_columns = [col for col in numeric_columns if col != target_column]

    # Ensure categorical_columns is provided
    if categorical_columns is None:
        categorical_columns = []

    # Remove explicitly defined categorical columns from numeric columns
    numeric_columns = [col for col in numeric_columns if col not in categorical_columns]

    augmented_data = []
    for _, row in df.iterrows():
        for _ in range(augmentation_factor):
            augmented_row = row.copy()

            # Augment numeric columns
            if numeric_columns:
                # Add Gaussian noise to numeric columns
                if "noise" in augmentation_methods:
                    noise = np.random.normal(0, 0.02, size=len(numeric_columns))  # Small noise
                    augmented_row[numeric_columns] += noise

                # Apply scaling to numeric columns
                if "scaling" in augmentation_methods:
                    scaling_factors = np.random.uniform(0.9, 1.1, size=len(numeric_columns))
                    augmented_row[numeric_columns] *= scaling_factors

            # Augment categorical columns
            if categorical_columns and "categorical_augmentation" in augmentation_methods:
                for col in categorical_columns:
                    if np.random.rand() < 0.1:  # Small chance of replacing
                        # Replace with a random value from the column's unique categories
                        categories = df[col].dropna().unique()
                        augmented_row[col] = np.random.choice(categories)

            augmented_row["Augmented"] = 1  # Mark as augmented
            augmented_data.append(augmented_row)

    # Create a DataFrame from augmented data
    augmented_df = pd.DataFrame(augmented_data)

    # Add 'Augmented' column for the original data
    df["Augmented"] = 0  # Mark original data

    # Combine original and augmented data
    return pd.concat([df, augmented_df], ignore_index=True)