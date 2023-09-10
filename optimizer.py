import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_season(date):
    month = pd.to_datetime(date).month
    if month in [12, 1, 2]:
        return 0.00 
    elif month in [3, 4, 5]:
        return 1.00 
    elif month in [6, 7, 8]:
        return 2.00 
    else:
        return 3.00
 
def configure(df):
    df = df.dropna(how='all')

    df.drop(df.loc[:, 'col1':'col47'].columns, axis=1, inplace=True)
    df.drop(df.loc[:, 'col49':'col64'].columns, axis=1, inplace=True)
    df.drop(df.loc[:, 'col85':'col100'].columns, axis=1, inplace=True)
    df.drop(df.loc[:, 'col130':'col136'].columns, axis=1, inplace=True)
    df.drop(df.loc[:, 'col167':'col168'].columns, axis=1, inplace=True)
    df.drop(df.loc[:, 'col223':'col228'].columns, axis=1, inplace=True)
    df.drop(df.loc[:, 'col237':'col244'].columns, axis=1, inplace=True)
    df.drop(df.loc[:, 'col457':'col472'].columns, axis=1, inplace=True)
    df.drop(df.loc[:, 'col513':'col576'].columns, axis=1, inplace=True)
    df.drop(df.loc[:, 'col609':'col616'].columns, axis=1, inplace=True)
    df.drop(df.loc[:, 'col793':'col800'].columns, axis=1, inplace=True)
    df.drop(df.loc[:, 'col1029':'col1052'].columns, axis=1, inplace=True)
    df.drop('col1454', axis=1, inplace=True)
    
    label_encoder = LabelEncoder()
    object_columns = df.select_dtypes(include=['object']).columns
    object_columns = object_columns[1:-2]
    df[object_columns] = df[object_columns].apply(lambda series: pd.Series(
        label_encoder.fit_transform(series[series.notnull()]),
        index=series[series.notnull()].index
    ))

    keys = label_encoder.classes_
    values = label_encoder.transform(label_encoder.classes_)
    dictionary = dict(zip(keys, values))

    df.drop(df.loc[:, 'col217':'col2198'].columns, axis=1, inplace=True)

    df.sort_values(by=['report_date'])
    df.sort_values(by=['client_id'])
    df['report_date'] = pd.to_datetime(df['report_date'])
    df.sort_values(by=['client_id', 'report_date'], inplace=True)
    df['last_report'] = df.groupby('client_id')['report_date'].diff().dt.days.fillna(0).astype(float)

    # add season
    df['season'] = df['report_date'].apply(get_season)

    return [df, dictionary] 
 
