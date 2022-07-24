# Fetch data from openml

# Misc
import numpy as np
import pandas as pd
# Sklearn
import sklearn
from sklearn.datasets import fetch_openml

def get_df(dataset):
    assert(dataset == 'credit-g')
    
    dataset = fetch_openml(dataset)
    target = 'target'
    df = pd.DataFrame(data= np.c_[dataset['data'], dataset[target]], columns= dataset['feature_names'] + [target])  
    
    # Renaming target for training later
    df[target] = df[target].apply(lambda x : 0.0 if x == 'bad' or x == 0.0 else 1)
    
    # Subsetting features to keep only continuous, discrete and ordered categorical
    feature_names = ['checking_status', 'duration', 'credit_amount',
                 'savings_status','employment','installment_commitment',
                 'residence_since','age','existing_credits','num_dependents',
                 'own_telephone','foreign_worker']
    
    df = df[feature_names + [target]]
    
    # Casting to float for later purpose
    return df, target, feature_names