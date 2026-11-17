import pandas as pd
from sklearn.model_selection import train_test_split

# Features used by the Random Forest model
FEATURE_COLUMNS = [
    "speed",
    "longitudinal_acc",
    "lateral_acc",
    "road_curvature"
]


def preprocess_and_engineer_features(df: pd.DataFrame):

    df = df.copy()

    # Handle missing values in numerical columns
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

        df[column] = df[column].fillna(
            df[column].median()
        )

    # Handle remaining invalid numerical values
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    return df


def split_by_journey(df: pd.DataFrame, test_size=0.2):

    if "journey_id" not in df.columns:
        raise ValueError(
            "Dataset must contain a journey_id column."
        )

    if "event_label" not in df.columns:
        raise ValueError(
            "Dataset must contain an event_label column."
        )

    journey_ids = df["journey_id"].dropna().unique()

    if len(journey_ids) < 2:
        raise ValueError(
            "At least two different journeys are required."
        )

    train_ids, test_ids = train_test_split(
        journey_ids,
        test_size=test_size,
        random_state=42
    )

    train_df = df[df["journey_id"].isin(train_ids)].copy()
    test_df = df[df["journey_id"].isin(test_ids)].copy()

    return train_df, test_df
