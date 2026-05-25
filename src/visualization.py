import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_target_distribution(
    df: pd.DataFrame,
    target_column: str = "income"
) -> None:
    """
    Plot target class distribution.
    """
    plt.figure(figsize=(8, 5))

    df[target_column].value_counts().plot(
        kind="bar"
    )

    plt.title("Income Class Distribution")
    plt.xlabel("Income Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)

    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numerical_features: list[str]
) -> None:
    """
    Plot correlation heatmap for numerical features.
    """
    correlation_matrix = df[numerical_features].corr()

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Matrix of Numerical Features")

    plt.show()


def plot_confusion_matrix(
    matrix,
    title: str,
    labels: list[str] | None = None
) -> None:
    """
    Plot confusion matrix.
    """
    if labels is None:
        labels = ["<=50K", ">50K"]

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()