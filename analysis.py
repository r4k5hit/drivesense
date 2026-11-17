
import joblib
import pandas as pd

from preprocessing import (
    preprocess_and_engineer_features,
    FEATURE_COLUMNS
)


def analyze_journey(
    journey_df: pd.DataFrame,
    model_path="models/drivesense_rf.pkl"
):

    # Load trained Random Forest model
    model = joblib.load(model_path)

    # Preprocess journey data
    processed = preprocess_and_engineer_features(journey_df)

    # Predict driving events
    processed["predicted_event"] = model.predict(
        processed[FEATURE_COLUMNS]
    )

    # Explain detected driving behaviour
    def explain_row(row):

        if (
            row["predicted_event"] == "Normal driving"
            and row["longitudinal_acc"] <= -2.5
            and row["road_curvature"] >= 0.45
        ):
            return "Safe pre-bend braking before sharp curve (4H ALLGRIP)"

        elif row["predicted_event"] == "Harsh braking":
            return "Strong deceleration on straight road — flagged for review"

        elif row["predicted_event"] == "Sharp-turn event":
            return "High lateral acceleration in sharp bend"

        elif row["predicted_event"] == "Rapid acceleration":
            return "Rapid forward acceleration above threshold"

        return "Normal driving behaviour"

    # Generate contextual explanations
    processed["context_explanation"] = processed.apply(
        explain_row,
        axis=1
    )

    return processed
