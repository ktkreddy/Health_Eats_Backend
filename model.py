import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def scaling(dataframe):
    scaler=StandardScaler()
    prep_data=scaler.fit_transform(dataframe.iloc[:,6:15].to_numpy())
    return prep_data,scaler

def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric='cosine',algorithm='brute')
    neigh.fit(prep_data)
    return neigh

def build_pipeline(neigh,scaler,params):
    transformer = FunctionTransformer(neigh.kneighbors,kw_args=params)
    pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])
    return pipeline

def extract_data(dataframe,ingredients,ingredients_to_avoid):
    extracted_data=dataframe.copy()
    extracted_data=extract_ingredient_filtered_data(extracted_data,ingredients,ingredients_to_avoid)
    return extracted_data
    
#def extract_ingredient_filtered_data(dataframe,ingredients):
 #   extracted_data=dataframe.copy()
  #  regex_string=''.join(map(lambda x:f'(?=.*{x})',ingredients))
   # extracted_data=extracted_data[extracted_data['RecipeIngredientParts'].str.contains(regex_string,regex=True,flags=re.IGNORECASE)]
    #return extracted_data

def extract_ingredient_filtered_data(dataframe, include_ingredients, ingredients_to_avoid):
    extracted_data = dataframe.copy()

    # Create regex for required ingredients
    include_regex = ''.join(map(lambda x: f'(?=.*{x})', include_ingredients))

    # Create regex for excluded ingredients
    exclude_regex = '|'.join(map(lambda x: f'(?=.*{x})', ingredients_to_avoid))

    # Filter to include required ingredients
    extracted_data = extracted_data[extracted_data['RecipeIngredientParts'].str.contains(include_regex, regex=True, flags=re.IGNORECASE)]

    # Filter out excluded ingredients
    extracted_data = extracted_data[~extracted_data['RecipeIngredientParts'].str.contains(exclude_regex, regex=True, flags=re.IGNORECASE)]

    return extracted_data


def apply_pipeline(pipeline,_input,extracted_data):
    _input=np.array(_input).reshape(1,-1)
    return extracted_data.iloc[pipeline.transform(_input)[0]]

def recommend(dataframe,_input,ingredients=[],ingredients_to_avoid=[],params={'n_neighbors':5,'return_distance':False}):
        extracted_data=extract_data(dataframe,ingredients,ingredients_to_avoid)
        if extracted_data.shape[0]>=params['n_neighbors']:
            prep_data,scaler=scaling(extracted_data)
            neigh=nn_predictor(prep_data)
            pipeline=build_pipeline(neigh,scaler,params)
            return apply_pipeline(pipeline,_input,extracted_data)
        else:
            return None

def extract_quoted_strings(s):
    # Find all the strings inside double quotes
    strings = re.findall(r'"([^"]*)"', s)
    # Join the strings with 'and'
    return strings

def output_recommended_recipes(dataframe):
    if dataframe is not None:
        output=dataframe.copy()
        output=output.to_dict("records")
        for recipe in output:
            recipe['RecipeIngredientParts']=extract_quoted_strings(recipe['RecipeIngredientParts'])
            recipe['RecipeInstructions']=extract_quoted_strings(recipe['RecipeInstructions'])
    else:
        output=None
    return output

