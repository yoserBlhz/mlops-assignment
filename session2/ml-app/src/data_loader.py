"""
Data loader module for Iris dataset
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Tuple, List


def load_iris_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split the Iris dataset

    Args:
        test_size: Proportion of dataset to include in test split
        random_state: Random seed for reproducibility

    Returns:
        tuple: X_train, X_test, y_train, y_test as numpy arrays
    """
    try:
        iris = load_iris()
        X = iris.data
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print("Successfully loaded Iris dataset")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Classes: {np.unique(y)}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Error loading Iris dataset: {e}")
        raise


def get_feature_names() -> List[str]:
    """
    Return feature names for the Iris dataset

    Returns:
        list: Feature names as strings
    """
    try:
        iris = load_iris()
        feature_names = iris.feature_names
        print(f"Feature names: {feature_names}")
        return feature_names
    except Exception as e:
        print(f"Error getting feature names: {e}")
        raise


def get_target_names() -> List[str]:
    """
    Return target names for the Iris dataset

    Returns:
        list: Target class names as strings
    """
    try:
        iris = load_iris()
        target_names = list(iris.target_names)
        print(f"Target names: {target_names}")
        return target_names
    except Exception as e:
        print(f"Error getting target names: {e}")
        raise


def load_iris_as_dataframe() -> pd.DataFrame:
    """
    Load Iris dataset as a pandas DataFrame for exploration

    Returns:
        pd.DataFrame: Iris dataset with features and target
    """
    try:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["target"] = iris.target
        df["species"] = df["target"].apply(lambda x: iris.target_names[x])

        print(f"Loaded Iris dataset as DataFrame with {len(df)} rows")
        return df

    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        raise


def get_dataset_info() -> dict:
    """
    Get comprehensive information about the Iris dataset

    Returns:
        dict: Dataset information
    """
    try:
        iris = load_iris()

        info = {
            "feature_names": iris.feature_names,
            "target_names": list(iris.target_names),
            "n_samples": iris.data.shape[0],
            "n_features": iris.data.shape[1],
            "n_classes": len(iris.target_names),
            "class_distribution": dict(
                zip(*np.unique(iris.target, return_counts=True))
            ),
        }

        print("Dataset Information:")
        print(f"Samples: {info['n_samples']}")
        print(f"Features: {info['n_features']}")
        print(f"Classes: {info['n_classes']}")
        print(f"Class distribution: {info['class_distribution']}")

        return info

    except Exception as e:
        print(f"Error getting dataset info: {e}")
        raise


if __name__ == "__main__":
    print("Testing data_loader module...")

    X_train, X_test, y_train, y_test = load_iris_data()
    features = get_feature_names()
    targets = get_target_names()
    df = load_iris_as_dataframe()
    print(f"DataFrame columns: {df.columns.tolist()}")
    info = get_dataset_info()

    print("All data_loader tests passed!")
