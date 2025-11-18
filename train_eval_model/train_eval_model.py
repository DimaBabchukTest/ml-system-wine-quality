
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

import pickle

def main():
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    # Data Preparation
    red_wines_file = "./wine_quality_data/winequality_red.csv"
    df_wine_red = pd.read_csv(red_wines_file, sep=";")
    white_wines_file = "./wine_quality_data/winequality_white.csv"
    df_wine_white = pd.read_csv(white_wines_file, sep=";")

    # Combines 2 data sets in one 
    full_data_frame = pd.concat([df_wine_red,df_wine_white],ignore_index=True)

    full_data_frame.columns = full_data_frame.columns.str.lower().str.replace(' ', '_')

    # Make target variable for qulity of wine -> quality> 5 -> Good (1) else Bad (0)  
    full_data_frame['good_wine'] = (full_data_frame['quality'] > 5).values.astype(int)
    # we drop 'quality' column 
    del full_data_frame['quality']

    # we drop duplicates rows 
    print("Before:", full_data_frame.shape)
    full_data_frame = full_data_frame.drop_duplicates().reset_index(drop=True)
    print("After:", full_data_frame.shape)

    all_columns_array = full_data_frame.columns.values

    target_column = 'good_wine'
    all_columns_but_target = all_columns_array[all_columns_array != target_column]

    # End of data Preparation
    # Data Split
    df_train_full, df_test = train_test_split(full_data_frame, 
                                            test_size=0.2, 
                                            random_state = RANDOM_SEED, 
                                            stratify=full_data_frame.good_wine,
                                            shuffle=True)

    df_train, df_val =  train_test_split(df_train_full, 
                                            test_size=0.25, 
                                            random_state = RANDOM_SEED, 
                                            stratify=df_train_full.good_wine,
                                            shuffle=True)


    X_test = df_test[all_columns_but_target]
    y_test = df_test[target_column]

    X_train =  df_train[all_columns_but_target]
    y_train = df_train[target_column]

    X_val = df_val[all_columns_but_target]
    y_val = df_val[target_column]

    # End of Data Split

    ## Train and calibrate model with final fetures
    final_features_without_target = [ 'volatile_acidity', 'citric_acid', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol' ]

    best_max_depth=30
    best_n_estimators=120
    best_min_samples_leaf=5
    best_min_samples_split=10

    X_train_final = X_train[final_features_without_target]
    X_val_final = X_val[final_features_without_target]
    print('<-Start training procedure ->')
    best_random_forest = RandomForestClassifier(
                                        max_depth=best_max_depth, 
                                        n_estimators=best_n_estimators,
                                        min_samples_leaf=best_min_samples_leaf,
                                        min_samples_split=best_min_samples_split,
                                        n_jobs=-1, 
                                        random_state=RANDOM_SEED)
    best_random_forest.fit(X_train_final, y_train)


    X_val_final = X_val[final_features_without_target]

    calibration_X, threshold_X, calibration_y, threshold_y = train_test_split(
        X_val_final,
        y_val,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=y_val,
        shuffle=True,
    )  

    calibrated_rf = CalibratedClassifierCV(
        best_random_forest,
        method="sigmoid", 
        cv="prefit"            # IMPORTANT: means RF is already trained
    )
    
    calibrated_rf.fit(calibration_X, calibration_y)

    # Let's find out the best treshould use validation data set.
    probs_val = calibrated_rf.predict_proba(threshold_X)[:,1]

    thresholds = np.linspace(0, 1, 101)
    best_t = 0
    best_f1 = 0

    for t in thresholds:
        preds = (probs_val >= t).astype(int)
        f1 = f1_score(threshold_y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(' Best F1 on threshold data set', best_f1)
    print(' Best treshould on threshold data set', best_t)

    ## Final check on test

    X_test_final = X_test[final_features_without_target]
    probs_val_final = calibrated_rf.predict_proba(X_test_final)[:,1]
    brier_score_test_calibrated = brier_score_loss(y_test, probs_val_final)


    print('Calibrated model ROC_AUC Score for test',  roc_auc_score(y_test, probs_val_final)) 
    print('Calibrated model F1 Score for test threshold = 0.50', f1_score(y_test, (probs_val_final >= 0.50).astype(int)))
    print('Calibrated model Brier Score for smallest features on test set', brier_score_test_calibrated)

    #store model to pick file
    model_file = './model_artifact/wine_rate_v1.bin'
    with open(model_file, 'wb') as f_out:
        pickle.dump(calibrated_rf, f_out)

    print('<-End training procedure ->')    

if __name__ == "__main__":
    main()        
