# imports

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import silhouette_score, classification_report, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

np.random.seed(42) # Set seed so that the results are reproducable
# This shoud be deleted or commented out in case we want a more stohastic algorithm

print('Loaded all Libraries')


# Create Optimul Cluster Function 
def find_optimal_clusters(df, algorithm=KMeans(), max_clusters=10):
    """
    Determine the optimal number of clusters using the Silhouette Score.

    Parameters:
        df (pd.DataFrame): Input dataset.
        algorithm (sklearn.cluster.BaseEstimator): Clustering algorithm to use. Defaults to KMeans().
        max_clusters (int, optional): Maximum number of clusters to consider. Defaults to 10.

    Returns:
        int: Optimal number of clusters for the given dataset.
    """

    # Initialize the optimal score and the optimal number of clusters
    best_score = -1
    optimal_n_clusters = None

    # Try out all possible numbers of clusters from 2 to max_clusters
    for n_clusters in range(2, max_clusters + 1):
        algorithm.n_clusters = n_clusters
        labels = algorithm.fit_predict(df)
        score = silhouette_score(df, labels)

        # Update the optimal score and the optimal number of clusters if necessary
        if score > best_score:
            best_score = score
            optimal_n_clusters = n_clusters

    return optimal_n_clusters

print('Optimal Cluster function Done')

# Create a function for creating embedings 

def create_embeding(ids, embeding_dims = 8, random = False, prefix=None):
    rows = len(set(churn.Customer_Id_No))
    
    if random:
        matrix = np.random.normal( size=(rows, embeding_dims) )
    else:
        matrix = np.arange(0, rows*embeding_dims).reshape(rows,embeding_dims)

    
    if prefix is not None:
        columns=list(map(lambda x: prefix+'V_'+str(x), range(embeding_dims))) 
    else:
        columns=list(map(lambda x: 'V_'+str(x), range(embeding_dims))) 
    

    matrix = pd.DataFrame(matrix[ids]
                          , columns=columns)
    
    
    return matrix

print('Embeding function done')


# Load Dataset 
churn = pd.read_csv('churn_data.csv'
                    , parse_dates=True)

print('Loaded Data')

# Factorize Customer Id and plan type

customer_id_factor = pd.factorize(churn.customer_id)
churn['Customer_Id_No'] = customer_id_factor[0]

plant_type_factor =  pd.factorize(churn.plan_type)
churn['plan_type_id'] =  plant_type_factor[0]

print('Factorization Done')

# Create variable that calculates the difference in days between issuing date and date

churn.issuing_date = pd.to_datetime(churn.issuing_date)
churn.date = pd.to_datetime(churn.date)
churn['no_days'] = churn.date-churn.issuing_date 
churn['no_days'] = churn.no_days.apply(lambda x: x.days)

print('no_days created')

# Create Customer ID embedings
df_c_v = create_embeding(ids=churn.Customer_Id_No, prefix='Customer_', random=False)
df = pd.concat([churn, df_c_v], axis=1)

print('Customer Id Embeding created')

# Create Plan Type Embeding 
df_p_v = create_embeding(ids=churn.plan_type_id, prefix='Plan_Type_', random=False)
df = pd.concat([df, df_p_v], axis=1)

print('Plan Type Embeding created')

# Drop Irelevent Columns 
df = df.drop( 'customer_id date plan_type issuing_date Customer_Id_No plan_type_id'.split(), axis=1 )

print('Droped irelevent Columns')

# Creat X and y 
X = df.drop('churn', axis=1)
y = df.churn

print('X and y created!')

# Perform Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

print('Train Test Split Done')

# Find the optimal number of clustes 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Scale the data

imputer = KNNImputer()
X_train_scaled = imputer.fit_transform(X_train_scaled) # Impute the missing values

optimal = find_optimal_clusters(df=X_train_scaled) # Store the optimal number of clusters

print(f'Optimal number of Clusters found. Optimal number is {optimal}')

# Create a clustering pipeline 

pipe_clustering = Pipeline(steps=[
    ( 'scaler', StandardScaler() )
    , ('imputer', KNNImputer())
    , ('kmeans', KMeans(n_clusters=optimal))

])

print('Clustering Pipeline Created')

# Create the cluster variable for the test and train set 

X_train['cluster'] = pipe_clustering.fit_predict(X_train)
X_test['cluster'] = pipe_clustering.predict(X_test)

print('Cluster variable created!')

# Create classification pipeline 
pipe = Pipeline(steps=[
    ( 'scaler', StandardScaler() )
    , ('imputer', KNNImputer())
    , ('estimator', RandomForestClassifier(n_estimators=1000))

])

print("Classification pipeline done")

# Crossvalidate the precision and recall of the pipeline
precision_score = cross_validate(estimator=pipe, X=X_train, y=y_train,cv=10, n_jobs=-1, scoring='precision')
recall_score = cross_validate(estimator=pipe, X=X_train, y=y_train,cv=10, n_jobs=-1, scoring='recall')

print("Crossvalidation Done")

print(f'Precision Score: {precision_score['test_score'].mean()}  , Recall Score: {recall_score['test_score'].mean()}')


# Fit the pipe to the train data

pipe.fit(X_train, y_train)

print('Pipe fited')

# Test the pipe
y_pred = pipe.predict(X_test)
class_report = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True) # Save it with all the 
print(classification_report(y_true=y_test, y_pred=y_pred, output_dict=False))

# Dump the metrics dictionary into a joblib file 
dump(value=class_report, filename='metrics.joblib')

print('Class Report dumped')

# Dump both pipelines
dump(value=pipe_clustering, filename='pipe_clustering.joblib')
dump(value=pipe, filename='pipe.joblib')

print('Dumped pipelines!')

# Create new CSV file with the predict column
X['cluster'] = pipe_clustering.predict(X=X)
X_predict = pipe.predict(X)
new_churn = pd.read_csv('churn_data.csv')
new_churn['predict'] = X_predict
new_churn.to_csv('churn_data_predictions.csv', index=False)


print( 'New CSV file Created' )