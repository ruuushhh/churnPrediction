import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """
    #Defining the map function
    def binary_map(feature):
        return feature.map({'Yes':1, 'No':0})

    # Encoding gender category
    df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})

    #Encoding the other categoric features with more than two categories
    df = pd.get_dummies(df, drop_first=True)
    

    #feature scaling
    sc = MinMaxScaler()
    df['Subscription_Length_Months'] = sc.fit_transform(df[['Subscription_Length_Months']])
    df['Monthly_Bill'] = sc.fit_transform(df[['Monthly_Bill']])
    df['Total_Usage_GB'] = sc.fit_transform(df[['Total_Usage_GB']])
    return df