import pandas as pd


# Save model experiment results
def save_results(results, filepath="results/model_results.csv"):

    df = pd.DataFrame(results)

    df.to_csv(filepath, index=False)

    return df


# Select the best model based on R2 score
def select_best_model(results_df):

    best_row = results_df.loc[results_df["R2"].idxmax()]

    best_model_name = best_row["Model"]

    return best_model_name