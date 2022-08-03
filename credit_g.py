

# takes in credit-g data without the target and converts all nominal features to numeric
def nominal_to_numeric(X):
    class_map = {
        'good' : 1,
        'bad' : 0
    }
    #X['class'] = X['class'].map(class_map)


    foreign_worker_map = {
        'yes' : 0,
        'no'  : 1
    }

    X['foreign_worker'] = X['foreign_worker'].map(foreign_worker_map)


    own_telephone_map = {
        'none' : 0,
        'yes' : 1
    }

    X['own_telephone'] = X['own_telephone'].map(own_telephone_map)


    job_map = {
        'unemp/unskilled non res' : 0,
        'unskilled resident' : 1,
        'skilled' : 2,
        'high qualif/self emp/mgmt' : 3
    }

    X['job'] = X['job'].map(job_map)


    housing_map = {
        'rent' : 0,
        'own' : 1,
        'for free' : 2
    } 

    X['housing'] = X['housing'].map(housing_map)


    other_payment_plans_map = {
        'bank' : 0,
        'stores' : 1,
        'none' : 2
    }

    X['other_payment_plans'] = X['other_payment_plans'].map(other_payment_plans_map)


    property_magnitude_map = {
        'real estate' : 0,
        'life insurance' : 1,
        'car' : 2,
        'no known property' : 3
    }

    X['property_magnitude'] = X['property_magnitude'].map(property_magnitude_map)


    other_parties_map = {
        'none' : 0,
        'co applicant' : 1,
        'guarantor' : 2
    }

    X['other_parties'] = X['other_parties'].map(other_parties_map)


    personal_status_map = {
        'male div/sep' : 0,
        'female div/dep/mar' : 1,
        'male single' : 2,
        'male mar/wid' : 3,
        'female single' : 4
    }

    X['personal_status'] = X['personal_status'].map(personal_status_map)


    employment_map = {
        'unemployed' : 0,
        '<1' : 1,
        '1<=X<4' : 2,
        '4<=X<7' : 3,
        '>=7' : 4
    }

    X['employment'] = X['employment'].map(employment_map)


    savings_status_map = {
        '<100' : 0,
        '100<=X<500' : 1,
        '500<=X<1000' : 2,
        '>=1000' : 3,
        'no known savings' : 4
    }

    X['savings_status'] = X['savings_status'].map(savings_status_map)


    checking_status_map = {
        '<0' : 0,
        '0<=X<200' : 1,
        '>=200' : 2,
        'no checking' : 3
    }

    X['checking_status'] = X['checking_status'].map(checking_status_map)


    credit_history_map = {
        'no credits/all paid' : 0,
        'all paid' : 1,
        'existing paid' : 2,
        'delayed previously' : 3,
        'critical/other existing credit' : 4
    }

    X['credit_history'] = X['credit_history'].map(credit_history_map)


    purpose_map = {
        'radio/tv': 0,
        'education' : 1,
        'furniture/equipment' : 2, 
        'new car' : 3,
        'used car' : 4,
        'business' : 5,
        'domestic appliance' : 6,
        'repairs' : 7,
        'vacation' : 8,
        'retraining' : 9,
        'other' : 10
    }
    
    X['purpose'] = X['purpose'].map(purpose_map)

    return X