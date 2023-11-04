import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Reader, Dataset



def get_city_df_recommendation(city_id, n):
    """
     Collaborative Filtering with Surprise Library: Extract features and use Cosine Similarity as the similarity metric
    """

    data = pd.read_csv("Tree_for_model_test2.csv")

    tfdif = TfidfVectorizer(stop_words='english')
    tfdif_matrix = tfdif.fit_transform(data['TreeName']) #city name and tree name return same value
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

    return top['TreeName'], tail['TreeName']

def get_city_cf_recommendation(city_id, n):

    """
     Content-Based Filtering with Scikit-Learn : Use the SVD algorithm to perform Collaborative Filtering
    """

    old_data = pd.read_csv("Tree_for_model_test2.csv")

    city_trees = old_data[old_data['IDCity'] == city_id]

    reader = Reader(rating_scale=(1,4))

    new_data = Dataset.load_from_df(city_trees[['IDCity','IDTreeSpecies','Suggested_Frequency_of_Planting']],reader)

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

    return top_n_trees,  bottom_n_trees


def get_city_hybrid_recommendation(city_id, n):
    """
    Hybrid Filtering Model: Vombine collaborative Filtering and Content-Based Filtering
    """

    first_recommendation : pd.DataFrame = get_city_cf_recommendation(city_id, n*2)

    second_recommendation : pd.DataFrame = get_city_df_recommendation(city_id, n*2)

    hybrid = pd.concat([first_recommendation,second_recommendation]).groupby(['TreeName'])['TreeName'].count().reset_index(name='count').sort_values(['count'],ascending=False)

    hybrid_top = hybrid.head(n)
    hybrid_bottom = hybrid.tail(n)

    return hybrid_top, hybrid_bottom
