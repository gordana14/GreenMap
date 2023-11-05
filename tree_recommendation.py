import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Reader, Dataset
from surprise.model_selection import cross_validate

import numpy as np

def get_city_df_recommendation(city_id, n, data):

    """
     Collaborative Filtering with Surprise Library: Extract features and use Cosine Similarity as the similarity metric

     Args:
        city_id (int) : Id of the city from data set
        n (int) : number of recommendation to return

    """

    #data = pd.read_csv("Tree_for_model_test2.csv")  ## Add data set here
    data = data.copy()
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

    return top[['IDTreeSpecies','TreeName']], tail[['IDTreeSpecies','TreeName']]

def get_city_cf_recommendation(city_id, n, data):

    """
     Content-Based Filtering with Scikit-Learn : Use the SVD algorithm to perform Collaborative Filtering

    Args:
        city_id (int) : Id of the city from data set
        n (int) : number of recommendation to return
    """

    old_data = data.copy() #

    city_trees = old_data[old_data['IDCity'] == city_id]

    reader = Reader(rating_scale=(0,129))

    new_data = Dataset.load_from_df(old_data[['IDCity','IDTreeSpecies','Rating']],reader) # ranking used based on suggested frequency of planting

    algo = SVD()

    cross_validate(algo,new_data, measures=['RMSE','MAE'], cv=5, verbose=True)
    trainset = new_data.build_full_trainset()

    algo.fit(trainset)
    
    tree_ids = old_data['IDTreeSpecies'].unique().tolist()


    predictions = []

    for tree_id in tree_ids:
        predictions.append((tree_id, algo.predict(city_id,tree_id).est))
    n_predicitons = sorted(predictions, key= lambda x : x[1], reverse= True)

    top_n_predictions = n_predicitons[:n]

    top_n_trees_id = [x[0] for x in top_n_predictions]
    top_n_trees_data = data[data['IDTreeSpecies'].isin(top_n_trees_id)]
    top_n_trees = top_n_trees_data.groupby(['IDTreeSpecies', 'TreeName'])['IDTreeSpecies'].count().reset_index(name='count').sort_values(['count'],ascending=False).head(n)


    bottom_n_predictions = n_predicitons[len(n_predicitons)-n:]
    bottom_n_trees_id = [x[0] for x in bottom_n_predictions]
    bottom_n_trees_data = data[data['IDTreeSpecies'].isin(bottom_n_trees_id)]

    bottom_n_trees = bottom_n_trees_data.groupby(['IDTreeSpecies', 'TreeName'])['IDTreeSpecies'].count().reset_index(name='count').sort_values(['count'],ascending=False).tail(n)


    return top_n_trees[['IDTreeSpecies','TreeName']], bottom_n_trees[['IDTreeSpecies','TreeName']]


def get_city_hybrid_recommendation(city_id, n, data):

    """
    Hybrid Filtering Model: Vombine collaborative Filtering and Content-Based Filtering

    Args:
        city_id (int) : Id of the city from data set
        n (int) : number of recommendation to return
    """

    first_recommendation, first_not = get_city_cf_recommendation(city_id, n*2, data.copy()) #collaborative Filtering

    second_recommendation, second_not  = get_city_df_recommendation(city_id, n*2, data.copy()) # Content-Based Filtering

    hybrid_recommned = pd.concat([first_recommendation,second_recommendation]).groupby(['IDTreeSpecies', 'TreeName'])['TreeName'].count().reset_index(name='count').sort_values(['count'],ascending=False)
    hybrid_bottom =  pd.concat([first_not,second_not]).groupby(['IDTreeSpecies', 'TreeName'])['TreeName'].count().reset_index(name='count').sort_values(['count'],ascending=False)

    hybrid_top = hybrid_recommned.head(n)
    hybrid_bottom = hybrid_bottom.head(n)

    return hybrid_top[['IDTreeSpecies','TreeName']], hybrid_bottom[['IDTreeSpecies','TreeName']]

if __name__ == '__main__':
    df= pd.read_csv("Tree_for_model_test.csv")
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
    res1_introduce, res1_reduce= get_city_df_recommendation(1, 10, df)
    res1_introduce = list(zip(res1_introduce.IDTreeSpecies,res1_introduce.TreeName))
    res1_reduce = list(zip(res1_reduce.IDTreeSpecies,res1_reduce.TreeName))

    # #second recc
    res2_introduce, res2_reduce = get_city_cf_recommendation(1, 10, df)
    res2_introduce = list(zip(res2_introduce.IDTreeSpecies,res2_introduce.TreeName))
    res2_reduce = list(zip(res2_reduce.IDTreeSpecies,res2_reduce.TreeName))

    #third recc
    res3_introduce, res3_reduce = get_city_hybrid_recommendation(1, 10, df)
    res3_introduce = list(zip(res3_introduce.IDTreeSpecies,res3_introduce.TreeName))
    res3_reduce = list(zip(res3_reduce.IDTreeSpecies,res3_reduce.TreeName))

    # print(res3)

    #print(res2)
    print("------------------Introduce Tree---------------------")
    print("\t similarty matrix \t\t ratig svd \t\t\t hybrid")
    for x,y,z in zip(res1_introduce,res2_introduce, res3_introduce):
         print(f"\t {x} \t {y} \t{z}")
    print("------------------Not recommended Tree---------------------")
    print("\t similarty matrix \t\t ratig svd \t\t\t hybrid")
    for x,y,z in zip(res1_reduce,res2_reduce, res3_reduce):
        print(f"\t {x} \t {y} \t{z}")

    #third recc

    # data=[]
    # f1 = 'C://Users/Goga/Documents/GitHub/GreenMap/Extanded.csv'
    # try:
    #     for index, row  in res2[2].iterrows():
    #         data.append({'IDTreeSpecies': row['IDTreeSpecies'], 'TreeName':row['TreeName']})
    #     file_export = pd.DataFrame(data)
    #     file_export.to_csv(f1, index=False) 
    # except:
    #     print("An exception occurred")

    #res1= get_city_df_recommendation(1, 10, df)
    #print(res1)
    #second recc
    #res2 = get_city_df_recommendation(1, 10, df)
    #third recc
    #res3 = get_city_hybrid_recommendation(1, 10, df)
    #data=[]
    #f1 = 'C://Users/Goga/Documents/GitHub/GreenMap/Extanded.csv'
    # try:
    #     for index, row  in res2[2].iterrows():
    #         data.append({'IDTreeSpecies': row['IDTreeSpecies'], 'TreeName':row['TreeName']})
    #     file_export = pd.DataFrame(data)
    #     file_export.to_csv(f1, index=False) 
    # except:
    #     print("An exception occurred")
    
    
    
    