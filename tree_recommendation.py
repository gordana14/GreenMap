import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Reader, Dataset

import numpy as np

def get_city_df_recommendation(city_id, n, data):

    """
     Collaborative Filtering with Surprise Library: Extract features and use Cosine Similarity as the similarity metric

     Args:
        city_id (int) : Id of the city from data set
        n (int) : number of recommendation to return

    """

    #data = pd.read_csv("Tree_for_model_test2.csv")  ## Add data set here

    tfdif = TfidfVectorizer(stop_words='english')
    tfdif_matrix = tfdif.fit_transform(data['Family']) #chose which column to be used to calcualte simmetry 
    cos_sim_matrix = cosine_similarity(tfdif_matrix,tfdif_matrix)

    city_trees = data[data['IDCity'] == city_id]
    recommended_trees = []

    for tree_id in city_trees['IDTreeSpecies']:

        tree_index = data[data['IDTreeSpecies'] == tree_id].index[0]

        tree_score = list(enumerate(cos_sim_matrix[tree_index]))

        sorted_trees = sorted(tree_score,key=lambda x : x[1],reverse=True)[1:n+1]

        recommended_trees += sorted_trees

    recommended_trees_ids = [x[0] for x in recommended_trees]

    recommended_trees_data = data[data['IDTreeSpecies'].isin(recommended_trees_ids)]

    n_trees = recommended_trees_data.groupby(['IDTreeSpecies', 'TreeName'])['IDTreeSpecies'].count().reset_index(name='count').sort_values(['count'],ascending=False)
    top = n_trees.head(n)
    tail = n_trees.tail(n)

    return top['TreeName'], tail['TreeName'], n_trees

def get_city_cf_recommendation(city_id, n, data):

    """
     Content-Based Filtering with Scikit-Learn : Use the SVD algorithm to perform Collaborative Filtering

    Args:
        city_id (int) : Id of the city from data set
        n (int) : number of recommendation to return
    """

    old_data = data #pd.read_csv("tree_for_model_test.csv") # add data set here

    city_trees = old_data[old_data['IDCity'] == city_id]

    reader = Reader(rating_scale=(0,129))

    new_data = Dataset.load_from_df(city_trees[['IDCity','IDTreeSpecies','Rating']],reader) # ranking used based on suggested frequency of planting

    algo = SVD()

    trainset = new_data.build_full_trainset()

    algo.fit(trainset)
    
    tree_ids = [old_data['IDTreeSpecies'].unique()]

    for tree_id in city_trees['IDTreeSpecies']:
        if tree_id in tree_ids:
            tree_ids.remove(tree_id)
    predictions = []

    for tree_id in tree_ids:
        predictions.append((tree_id, algo.predict(city_id,tree_id).est))

    n_predicitons = sorted(predictions, key= lambda x : x[1], reverse= True)
    
    top_n_predictions = n_predicitons[:n]
    top_n_trees_id = [x[0] for x in top_n_predictions]
    top_n_trees = old_data[old_data['IDTreeSpecies'].isin(top_n_trees_id)]

    bottom_n_predictions = n_predicitons[len(n_predicitons)-n:]
    bottom_n_trees_id = [x[0] for x in bottom_n_predictions]
    bottom_n_trees = old_data[old_data['IDTreeSpecies'].isin(bottom_n_trees_id)]

    return top_n_trees,  bottom_n_trees , n_predicitons


def get_city_hybrid_recommendation(city_id, n, data):

    """
    Hybrid Filtering Model: Vombine collaborative Filtering and Content-Based Filtering

    Args:
        city_id (int) : Id of the city from data set
        n (int) : number of recommendation to return
    """

    first_recommendation : pd.DataFrame = get_city_cf_recommendation(city_id, n*2, data) #collaborative Filtering

    second_recommendation : pd.DataFrame = get_city_df_recommendation(city_id, n*2, data) # Content-Based Filtering

    hybrid = pd.concat([first_recommendation,second_recommendation]).groupby(['TreeName'])['TreeName'].count().reset_index(name='count').sort_values(['count'],ascending=False)

    hybrid_top = hybrid.head(n)
    hybrid_bottom = hybrid.tail(n)

    return hybrid_top, hybrid_bottom

if __name__ == '__main__':
    df= pd.read_csv("C://Users/Goga/Documents/GitHub/GreenMap/Tree_for_model_test.csv")
    #check null
    df['Uses'].isnull().sum()
    df.isnull().sum().sum()
    df.loc[df['Uses'].isnull()]
    df['Uses'] = df['Uses'].replace(np.nan, 'NotAvailable')
    df['Suggested_Frequency_of_Planting'] = df['Suggested_Frequency_of_Planting'].replace(np.nan, 'NotAvailable')
    df.fillna('NotAvailable')
    #encoding
    df['Suggested_Frequency_of_Planting']= df['Suggested_Frequency_of_Planting'].map({'NotAvailable':1, 'Sparingly':2,'Moderately':3, 'Frequently':4})
    df['Rating'] =  np.ceil ((df['Suggested_Frequency_of_Planting'] * df['Tree_number_city'])/df['area'] )
    #p = df.groupby('Rating')['Rating'].agg(['count'])
    #first_recommendation
    res1= get_city_df_recommendation(1, 10, df)
    #print(res1)
    #second recc
    res2 = get_city_df_recommendation(1, 10, df)
    #third recc
    #res3 = get_city_hybrid_recommendation(1, 10, df)
    data=[]
    f1 = 'C://Users/Goga/Documents/GitHub/GreenMap/Extanded.csv'
    try:
        for index, row  in res2[2].iterrows():
            data.append({'IDTreeSpecies': row['IDTreeSpecies'], 'TreeName':row['TreeName']})
        file_export = pd.DataFrame(data)
        file_export.to_csv(f1, index=False) 
    except:
        print("An exception occurred")
    
    
    
    