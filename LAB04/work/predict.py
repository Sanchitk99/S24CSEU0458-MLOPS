# Predict one delivery, from the command line.

import pandas as pd
from pathlib import Path

from delivery import load_model


def main():
    work = Path(__file__).resolve().parent
    model = load_model(work / "model.joblib")

    data = pd.DataFrame([{
        "distance_km": 7,
        "prep_time_min": 25,
        "traffic_level": 3,
        "rain": 0
    }])

    prediction = model.predict(data)[0]

    print(f"PREDICTION: {prediction:.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
