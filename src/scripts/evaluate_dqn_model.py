import pandas as pd
import numpy as np
import random
import os
import sys
sys.path.append('..')
import warnings
from modules import constants, utils
import argparse
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Performance arguments')
    parser.add_argument('-e', '--eval_type', help='test or validation', default='test')
    parser.add_argument('-f', '--model_name', help='filename of the model. It should be in the models folder', default='dueling_dqn_per_63_10000000')
    parser.add_argument('-s', '--save_test_df', help ='yes or no', default='no')  
    args = parser.parse_args()

    print(args)

    if args.eval_type == 'test':
        test_df = pd.read_csv('../../data/test_set_constant.csv')
    elif args.eval_type == 'validation':
        test_df = pd.read_csv('../../data/val_set_constant.csv')
    else:
        raise ValueError('Unknown evaluation type. Please enter either "validation" or "test"')

    X_test = test_df.iloc[:, 0:-1]
    y_test = test_df.iloc[:, -1]

    X_test, y_test = np.array(X_test), np.array(y_test)

    try:
        dqn_model = utils.load_dqn(f'../../models/{args.model_name}.pkl') # seed, steps, and name of model should be arguments
    except:
        print('This model does not exist. Please use the filename of an existing model')
    dqn_test_df = utils.evaluate_dqn(dqn_model, X_test, y_test)
    if args.save_test_df == 'yes': #is an arg
        print(f'save_test_df: {args.save_test_df}')
        print(args)
        dqn_test_df.to_csv(f'../../test_dfs/test_df_{args.model_name}.csv', index=False)
    dqn_acc, dqn_f1, dqn_roc_auc = utils.test(dqn_test_df['y_actual'], dqn_test_df['y_pred'])
    dqn_avg_length = utils.get_avg_length(dqn_test_df)
    print(f'RESULTS FOR {args.model_name}')
    print(f'acc:{dqn_acc}, f1:{dqn_f1}, roc_auc:{dqn_roc_auc}, mean_episode_length:{dqn_avg_length}')
    
    