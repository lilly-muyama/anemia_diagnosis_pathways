import pandas as pd
import numpy as np
import random 
import os
import sys
sys.path.append('..')
import tensorflow as tf
from modules import utils, constants
import argparse

SEED = constants.SEED
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)
tf.set_random_seed(constants.SEED)
tf.compat.v1.set_random_seed(constants.SEED)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Parameters for dqn model')
    parser.add_argument('-f', '--train_set_name', help='Name of the training dataset e.g. train_set_basic', default = 'train_set_basic')
    parser.add_argument('-m', '--model_type', help='dqn or ddqn or dueling_dqn or dueling_ddqn', default='dueling_ddqn')
    parser.add_argument('-p', '--prioritized_replay', help ='yes or no', default='yes')  
    parser.add_argument('-t', '--steps', help='Number of timetseps to train the model', default=100000, type=int)
    args = parser.parse_args()

    if args.prioritized_replay == 'yes':
        per = True
    elif args.prioritized_replay == 'no':
        per = False
    else:
        raise ValueError('Unknown prioritized replay argument. Enter either yes or no.')

    train_df = pd.read_csv(f'../../data/{args.train_set_name}.csv')
    train_df = train_df.fillna(-1)

    X_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]

    X_train, y_train = np.array(X_train), np.array(y_train)

    model_name =f'{args.model_type}_{args.train_set_name[len("train_set_"):]}_{SEED}_{args.steps}' 
    if args.steps >1000000:
        print(f'Training {args.model_type} model over {args.steps} steps. This may take a while')
    else:
        print(f'Training {args.model_type} model over {args.steps} steps')
    if args.model_type == 'dqn':
        dqn_model = utils.stable_vanilla_dqn(X_train, y_train, args.steps, True, f'../../models/{model_name}', per)
    elif args.model_type == 'ddqn':
        dqn_model = utils.stable_double_dqn(X_train, y_train, args.steps, True, f'../../models/{model_name}', per)
    elif args.model_type == 'dueling_dqn':
        dqn_model = utils.stable_dueling_dqn(X_train, y_train, args.steps, True, f'../../models/{model_name}', per)
    elif args.model_type == 'dueling_ddqn':
        dqn_model = utils.stable_dueling_ddqn(X_train, y_train, args.steps, True, f'../../models/{model_name}', per)
    else:
        raise ValueError('Unknown model type. Enter "dqn" or "ddqn" or "dueling_dqn" or "dueling_ddqn"')

    print('Training complete and model saved. Please check in the models folder.')
    