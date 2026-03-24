import re
import numpy as np
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline


def scaling(dataframe):
    scaler = StandardScaler()
    prep_data = scaler.fit_transform(dataframe.iloc[:, 6:15].to_numpy())
    return prep_data, scaler


def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric="cosine", algorithm="brute")
    neigh.fit(prep_data)
    return neigh


def build_pipeline(neigh, scaler, params):
    transformer = FunctionTransformer(neigh.kneighbors, kw_args=params)
    return Pipeline([("std_scaler", scaler), ("NN", transformer)])


def extract_ingredient_filtered_data(dataframe, include_ingredients, ingredients_to_avoid):
    extracted_data = dataframe.copy()
    include_list = [x for x in (include_ingredients or []) if x and str(x).strip()]
    if include_list:
        include_regex = "".join(f"(?=.*{re.escape(str(x))})" for x in include_list)
        extracted_data = extracted_data[
            extracted_data["RecipeIngredientParts"].str.contains(include_regex, regex=True, flags=re.IGNORECASE)
        ]
    avoid_list = [x for x in (ingredients_to_avoid or []) if x and str(x).strip()]
    if avoid_list:
        exclude_regex = "|".join(f"(?=.*{re.escape(str(x))})" for x in avoid_list)
        extracted_data = extracted_data[
            ~extracted_data["RecipeIngredientParts"].str.contains(exclude_regex, regex=True, flags=re.IGNORECASE)
        ]
    return extracted_data


def apply_pipeline(pipeline, _input, extracted_data):
    _input = np.array(_input).reshape(1, -1)
    return extracted_data.iloc[pipeline.transform(_input)[0]]


def recommend(dataframe, _input, ingredients=None, ingredients_to_avoid=None, params=None):
    if params is None:
        params = {"n_neighbors": 5, "return_distance": False}
    extracted_data = extract_ingredient_filtered_data(dataframe, ingredients or [], ingredients_to_avoid or [])
    if extracted_data.shape[0] >= params["n_neighbors"]:
        prep_data, scaler = scaling(extracted_data)
        neigh = nn_predictor(prep_data)
        pipeline = build_pipeline(neigh, scaler, params)
        return apply_pipeline(pipeline, _input, extracted_data)
    return None


def extract_quoted_strings(s):
    return re.findall(r'"([^"]*)"', s)


def output_recommended_recipes(dataframe):
    if dataframe is None:
        return None
    output = dataframe.copy().to_dict("records")
    for recipe in output:
        recipe["RecipeIngredientParts"] = extract_quoted_strings(recipe["RecipeIngredientParts"])
        recipe["RecipeInstructions"] = extract_quoted_strings(recipe["RecipeInstructions"])
    return output
