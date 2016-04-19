import pandas as pd
import numpy as np

def data_transformer(df):

    df.drop('sample_uuid',axis=1,inplace=True)

    yes_no = {'yes': True, 'no': False}
    df['housing'] = df['housing'].map(yes_no)
    df['loan'] = df['loan'].map(yes_no)

    default = {'yes': 0, 'no': 1,'unknown': -1}
    df['default'] = df['default'].map(default)

    col_names = ['job', 'marital','education','contact']
    #pd.get_dummies(df[col_names])
    df = pd.concat([df, pd.get_dummies(df[col_names])], axis=1)
    df.drop(col_names,axis=1,inplace=True)

    poutcome = {'failure': 0, 'success': 1,'nonexistent': -1}
    df['poutcome'] = df['poutcome'].map(poutcome)

    df.drop(['month','day_of_week'],axis=1,inplace=True)
    df['housing'] = df['housing'].astype('bool')
    df['loan'] = df['loan'].astype('bool')
    return df
