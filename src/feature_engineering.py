import pandas as pd


# Create new features to improve model performance
def create_features(df):

    # Tourism growth rate
    df["tourism_growth_rate"] = df.groupby("Country")[
        "Inbound-Total Arrival(overnight stay)( In thousands)"
    ].pct_change()

    # Expenditure per tourist
    df["expenditure_per_tourist"] = (
        df["Inbound- Expenditure(Travel)(US$ Millions)"]
        / df["Inbound-Total Arrival(overnight stay)( In thousands)"]
    )

    # Replace infinite or missing values created during feature engineering
    df = df.replace([float("inf"), -float("inf")], pd.NA)
    df = df.fillna(0)

    return df