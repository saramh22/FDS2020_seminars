import dask.dataframe as dd
from glob import glob
import os
from dask import compute, persist
from dask import delayed

class preprocess():
    def __init__(self):
        print ("Cleaning of dataset is started")
    def cleaning(self):
        cols = ['Year', 'Month', 'DayOfWeek', 'Distance',
                'DepDelay', 'CRSDepTime', 'UniqueCarrier', 'Origin', 'Dest']

    # Create the dataframe
        df = dd.read_csv(sorted(glob(os.path.join('data', 'nycflights', '*.csv'))), usecols=cols,
                storage_options={'anon': True})

        df = df.sample(frac=0.2) # we blow out ram otherwise

        label = (df.DepDelay.fillna(16) > 15)

        df['CRSDepTime'] = df['CRSDepTime'].clip(upper=2399)
        del df['DepDelay']

        df, label = persist(df, label)
        df2 = dd.get_dummies(df.categorize()).persist()
        X_train, X_test = df2.random_split([0.9, 0.1], random_state=1234)
        y_train, y_test = label.random_split([0.9, 0.1], random_state=1234)

        return X_train, X_test, y_train, y_test
