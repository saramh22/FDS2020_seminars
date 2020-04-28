from dask.distributed import Client, LocalCluster
import joblib
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import preprocess as ps
import time
import dask_xgboost as dxgb

def main():
    object = ps.preprocess()
    X_train, X_test, y_train, y_test = object.cleaning()
    param_grid = {
                'objective': ['binary:logistic'],
                'nround': [1000],
                'max_depth': [8]
    }
    estimator = dxgb.XGBRegressor()
    grid_search = GridSearchCV(estimator, param_grid, verbose=2, cv=2,  n_jobs=-1)
    client = Client(processes=False)
    start_time = time.time()
    with joblib.parallel_backend("dask"):
        grid_search.fit(X_train, y_train)
    end_time = time.time()
    grid_search.predict(X_test)
    print ("time difference in GridSearchCV second XGBRegressor is %d seconds " % end_time)
    client.shutdown()
if __name__ == '__main__':
    main()
