# Fetch data from openml

# Misc
import numpy as np
import pandas as pd
# Sklearn
import sklearn
from sklearn.datasets import fetch_openml

from credit_g import nominal_to_numeric

def get_df(dataset):
    assert(dataset == 'credit-g')
    
    dataset = fetch_openml(dataset)
    target = 'target'
    print(len(dataset['data'].columns))

    # converts nominal features to numeric
    dataset['data'] = nominal_to_numeric(dataset['data'])
    
    df = pd.DataFrame(data= np.c_[dataset['data'], dataset[target]], columns= dataset['feature_names'] + [target])  
    
    # Renaming target for training later
    df[target] = df[target].apply(lambda x : 0.0 if x == 'bad' or x == 0.0 else 1)
    
    feature_names = ['checking_status',
                        'duration',
                        'credit_history',
                        'purpose',
                        'credit_amount',
                        'savings_status',
                        'employment',
                        'installment_commitment',
                        'personal_status',
                        'other_parties',
                        'residence_since',
                        'property_magnitude',
                        'age',
                        'other_payment_plans',
                        'housing',
                        'existing_credits',
                        'job',
                        'num_dependents',
                        'own_telephone',
                        'foreign_worker']
    
    df = df[feature_names + [target]]
    # Casting to float for later purpose
    df = df.astype(float)
    return df, target, feature_names