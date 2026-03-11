from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Initialize machine learning models
def get_models():

    models = {

        "Linear_Regression": LinearRegression(),

        "Random_Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),

        "Gradient_Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

    }

    return models