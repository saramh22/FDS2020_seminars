from dask.distributed import Client, LocalCluster
import preprocess as ps
import dask_xgboost as dxgb
import time
from sklearn.metrics import roc_auc_score, roc_curve

def main():
    object = ps.preprocess()
    X_train, X_test, y_train, y_test = object.cleaning()
    params = {'objective': 'binary:logistic',
          'max_depth': 8, 'eta': 0.01, 'subsample': 0.5,
          'min_child_weight': 1}
    print ("Start training dxgb")

    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)
    start_time = time.time()
    bst = dxgb.train(client, params, X_train, y_train)
    end_time = time.time()
    #time difference in dXGB is 1588108665
    print ("time difference in dXGB is %d seonds" % end_time)
    predictions = dxgb.predict(client, bst, X_test)
    #Accuracy = 0.6968888393419537
    print ("Accuracy score is : ")
    print(roc_auc_score(y_test.compute(),
                    predictions.compute()))
    client.shutdown()

if __name__ == '__main__':
    main()
