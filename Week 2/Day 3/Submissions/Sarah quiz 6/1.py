import preprocess as ps
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
import time

def main():
    object = ps.preprocess()
    X_train, X_test, y_train, y_test = object.cleaning()

    print ("First five columns -----")
    print (X_train.head())
    start_time = time.time()
    rg = xgb.XGBRegressor().fit(X_train, y_train)
    end_time = time.time()
    #time difference in normal XGBRegressor is 1588108665
    print ("time difference in normal XGBRegressor is %d seconds " % end_time)
    y_pred = rg.predict(X_test)
    #Accuracy = 0.7270127520153759
    print ("Accuracy score is : ")
    print(roc_auc_score(y_test.compute(),
                    y_pred))

if __name__ == '__main__':
    main()
