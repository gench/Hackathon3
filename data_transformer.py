import pandas as pd
import numpy as np

def data_transformer(df):

    dfout = df.drop('sample_uuid',axis=1,inplace=False)
    dfout.drop('duration', axis=1, inplace=True)

    yes_no = {'yes': True, 'no': False}
    dfout['housing'] = dfout['housing'].map(yes_no)
    dfout['loan'] = dfout['loan'].map(yes_no)

    default = {'yes': 0, 'no': 1,'unknown': -1}
    dfout['default'] = dfout['default'].map(default)

    col_names = ['job', 'marital','education','contact']
    dfout['job_admin.'] = dfout['job'].values[0] == 'admin.'
    dfout['job_blue-collar'] = dfout['job'].values[0] == 'blue-collar'
    dfout['job_entrepreneur'] = dfout['job'].values[0] == 'entrepreneur'
    dfout['job_housemaid'] = dfout['job'].values[0] == 'housemaid'
    dfout['job_management'] = dfout['job'].values[0] == 'management'
    dfout['job_retired'] = dfout['job'].values[0] == 'retired'
    dfout['job_self-employed'] = dfout['job'].values[0] == 'self-employed'
    dfout['job_services'] = dfout['job'].values[0] == 'services'
    dfout['job_student'] = dfout['job'].values[0] == 'student'
    dfout['job_technician'] =  dfout['job'].values[0] == 'technician'
    dfout['job_unemployed'] = dfout['job'].values[0] == 'unemployed'
    dfout['job_unknown'] = dfout['job'].values[0] == 'unknown'
    dfout['marital_divorced'] = dfout['marital'][0] == 'divorced'
    dfout['marital_married'] = dfout['marital'][0] == 'married'
    dfout['marital_single'] = dfout['marital'][0] == 'single'
    dfout['marital_unknown'] = dfout['marital'][0] == 'unknown'
    dfout['education_basic.4y'] = dfout['education'][0] == 'basic.4y'
    dfout['education_basic.6y'] = dfout['education'][0] == 'basic.6y'
    dfout['education_basic.9y'] = dfout['education'][0] == 'basic.9y'
    dfout['education_high.school'] = dfout['education'][0] == 'high.school'
    dfout['education_illiterate'] = dfout['education'][0] == 'illiterate'
    dfout['education_professional.course'] = dfout['education'][0] == 'professional.course'
    dfout['education_university.degree'] = dfout['education'][0] == 'university.degree'
    dfout['education_unknown'] = dfout['education'][0] == 'unknown'
    dfout['contact_cellular'] = dfout['contact'][0] == 'cellular'
    dfout['contact_telephone'] = dfout['contact'][0] == 'telephone'
    #dfout = pd.concat([dfout, pd.get_dummies(dfout[col_names])], axis=1)
    dfout.drop(col_names,axis=1,inplace=True)

    poutcome = {'failure': 0, 'success': 1,'nonexistent': -1}
    dfout['poutcome'] = dfout['poutcome'].map(poutcome)

    dfout.drop(['month','day_of_week'],axis=1,inplace=True)
    dfout['housing'] = dfout['housing'].astype('bool')
    dfout['loan'] = dfout['loan'].astype('bool')
    return dfout
