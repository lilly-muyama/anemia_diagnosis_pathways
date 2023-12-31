import pandas as pd
import ast
import sys
sys.path.append('..')
from modules import utils, constants
import argparse


if __name__== '__main__':
    parser = argparse.ArgumentParser(description = 'Parameters for dqn model')
    parser.add_argument('-f', '--filename', help='The filename of the saved csv file generated by the dqn model after testing', default='test_df_dueling_dqn_per_63_10000000')
    parser.add_argument('-l', '--class_list', help='List of classes to include on pathway diagram e.g. "[0, 3, 5]"', default=str(list(constants.CLASS_DICT.values())))
    parser.add_argument('-t', '--title', help='Title of the diagram', default='')
    parser.add_argument('-s', '--save', help='Whether to save daigram or not. yes or no', default='yes')
    parser.add_argument('-v', '--save_name', help='Filename for the saved diagram', default=None)

    args = parser.parse_args()


    dqn_test_df = pd.read_csv(f'../../test_dfs/{args.filename}.csv')
    try:
        class_list = ast.literal_eval(args.class_list)
    except:
        print('Enter the list in quotes')

    if args.save == 'yes':
        save = True
    else:
        save = False

    if args.save_name is None:
        args.save_name = f'{args.filename}_{args.class_list}'


    utils.create_sankey(dqn_test_df, class_list, args.title, save=save, filename=args.save_name)

    print('Pathway diagram has been saved...')


