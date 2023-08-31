import pandas as pd
import numpy as np
import argparse
import sys
sys.path.append('..')
from modules import utils


if __name__== '__main__':
    parser = argparse.ArgumentParser(description = 'Parameters for SOTA algorithm')
    parser.add_argument('-f', '--train_set_name', help='Name of the training dataset e.g. train_set_basic', default = 'train_set_basic')
    parser.add_argument('-m', '--model_type', help='dt or rf or xgb or svm', default='dt')
    args = parser.parse_args()

    is_svm=False
    mmc = None
    train_df = pd.read_csv(f'../../data/{args.train_set_name}.csv')
    train_df = train_df.fillna(-1)

    test_df = pd.read_csv(f'../../data/test_set_constant.csv')

    X_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]

    X_test = test_df.iloc[:, 0:-1]
    y_test = test_df.iloc[:, -1]

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    if args.model_type == 'rf':
        model = utils.rf(X_train, y_train)
    elif args.model_type == 'dt':
        model = utils.dt(X_train, y_train)
    elif args.model_type=='svm':
        model, mmc = utils.svm(X_train, y_train)
        is_svm = True
    elif args.model_type == 'xgb':
        model = utils.xgb(X_train, y_train)
    else:
        raise ValueError('Unknown model type. Use either rf, dt, xgb or svm')
    
    sota_test_df = utils.evaluate_sota_model(model, X_test, y_test, is_svm, mmc)

    acc, f1, roc_auc = utils.test(sota_test_df['y_actual'], sota_test_df['y_pred'])
    print(f'RESULTS FOR {args.model_type.upper()} TRAINED USING {args.train_set_name}')
    print(f'acc:{acc}, f1:{f1}, roc_auc:{roc_auc}')