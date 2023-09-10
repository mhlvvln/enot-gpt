import pandas as pd
import os
import sys
import numpy as np
from catboost import CatBoostClassifier, Pool

from optimizer import configure


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        result = sys.argv[2]
        if not filename.endswith(".csv"):
            print('only .csv files')
            return
        if not os.path.exists(filename):
            print('file is not exists')
            return

        file = pd.read_csv(filename, sep=";", low_memory=False)

        df, dictionary = configure(file)

        model = CatBoostClassifier()
        model.load_model('model_name (12)')

        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        new_data_pool = Pool(df, cat_features=categorical_features)
        predictions = model.predict_proba(new_data_pool)[:, 1]

        result_df = pd.read_csv(result, sep=";", low_memory=False)
        result_df['target'] = np.round(predictions).astype(int)
        result_df.to_csv("_byanya" + result, sep=';', index=False)

        print('file updated')
    else:
        print('using analyzer.py -filename')
        return


main()
