from stable_baselines import DQN
from modules.env import AnemiaEnv
from modules import constants
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
# from stable_baselines import bench, logger
# from stable_baselines.common.callbacks import CheckpointCallback
# import tensorflow

# tensorflow.set_random_seed(constants.SEED)


################################################################### DQN FUNCTIONS #####################################################################

def load_dqn(filename, env=None):
    '''
    Loads a previously saved DQN model
    '''
    model = DQN.load(filename, env=env)
    return model

def create_env(X, y, random=True):
    '''
    Creates and environment using the given data
    '''
    env = AnemiaEnv(X, y, random)
    return env

def stable_vanilla_dqn(X_train, y_train, timesteps, save=False, filename=None, per=False):
    '''
    Creates and trains a standard DQN model
    '''
    training_env = create_env(X_train, y_train)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1, policy_kwargs=dict(dueling=False),
                double_q=False, prioritized_replay=per)
    model.learn(total_timesteps=timesteps, log_interval=50000)
    if save:
        model.save(f'{filename}.pkl')
    training_env.close()
    return model

def stable_dueling_dqn(X_train, y_train, timesteps, save=False, filename=None, per=False):
    '''
    Creates and trains a dueling DQN model
    '''
    training_env = create_env(X_train, y_train)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1, double_q=False, prioritized_replay=per)
    model.learn(total_timesteps=timesteps, log_interval=10000)
    if save:
        model.save(f'{filename}.pkl')
    training_env.close()
    return model

def stable_dueling_ddqn(X_train, y_train, timesteps, save=False, filename=None, per=False):
    '''
    Creates and trains a dueling double DQN model
    '''
    training_env = create_env(X_train, y_train)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1, prioritized_replay=per)
    model.learn(total_timesteps=timesteps, log_interval=10000)
    if save:
        model.save(f'{filename}.pkl')
    training_env.close()
    return model

def stable_double_dqn(X_train, y_train, timesteps, save=False, filename=None, per=False):
    '''
    Creates and trains a double DQN model
    '''
    training_env = create_env(X_train, y_train)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1, policy_kwargs=dict(dueling=False), 
                prioritized_replay=per)
    model.learn(total_timesteps=timesteps, log_interval=10000)
    if save:
        model.save(f'{filename}.pkl')
    training_env.close()
    return model

def evaluate_dqn(dqn_model, X_test, y_test):
    '''
    Evaluates a DQN model on test data
    '''
    test_df = pd.DataFrame()
    env = create_env(X_test, y_test, random=False)
    count=0

    try:
        while True:
            count+=1
            obs, done = env.reset(), False
            while not done:
                action, _states = dqn_model.predict(obs, deterministic=True)
                obs, rew, done, info = env.step(action)
                if done == True:
                    test_df = test_df.append(info, ignore_index=True)
    except StopIteration:
        print('Testing done.....')
    return test_df

############################################################### OTHER MODELS ###################################################################

def rf(X_train, y_train):
    '''
    Creates and trains a random forest model
    '''
    rf = RandomForestClassifier(random_state=constants.SEED).fit(X_train, y_train)
    return rf

def dt(X_train, y_train):
    '''
    Creates and trains a decision tree model
    '''
    dt = DecisionTreeClassifier(criterion='entropy', random_state=constants.SEED).fit(X_train, y_train)
    return dt


def xgb(X_train, y_train):
    '''
    Creates and trains an XGBoost model
    '''
    xg = xgboost.XGBClassifier(random_state=constants.SEED).fit(X_train, y_train)
    return xg

def svm(X_train, y_train):
    '''
    Creates and trains an SVM standard DQN model
    '''
    mmc = MinMaxScaler()
    X_train_norm = mmc.fit_transform(X_train)
    svm_model = SVC(kernel='poly', C=100, decision_function_shape='ovo', random_state=constants.SEED).fit(X_train_norm, y_train)
    return svm_model, mmc

def evaluate_sota_model(model, X_test, y_test, is_svm=False, mmc=None):
    '''
    Evaluates a state-of-the-art model
    '''
    if (is_svm==True)  & (mmc is None):
        print('MMC can\'t be None for SVM')
        return
    elif is_svm:
        X_test = mmc.fit_transform(X_test)

    y_pred = model.predict(X_test)
    test_df = pd.DataFrame()
    test_df['y_actual'] = y_test
    test_df['y_pred'] = y_pred
    return test_df



############################################################ PERFROMANCE FUNCTIONS ##############################################################

def multiclass(actual_class, pred_class, average = 'macro'):
    '''
    Returns the ROC-AUC score for multi-labeled data
    '''

    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    avg = sum(roc_auc_dict.values()) / len(roc_auc_dict)
    return avg

def test(ytest, ypred):
    '''
    Return perfromance metrics for a model
    '''
    acc = accuracy_score(ytest, ypred)
    f1 = f1_score(ytest, ypred, average ='macro', labels=np.unique(ytest))
    try:
        roc_auc = multiclass(ytest, ypred)
    except:
        roc_auc = None
    return acc*100, f1*100, roc_auc*100

def get_avg_length(df):
    '''
    Return the average length of individual pathways 
    '''
    length = np.mean(df.episode_length)
    return length

def plot_confusion_matrix(y_actual, y_pred, save=False, filename=False):
    '''
    Plots a confusion matrix of a model's results
    '''
    cm = confusion_matrix(y_actual, y_pred)
    cm_df = pd.DataFrame(cm, index = constants.CLASS_DICT.keys(), columns = constants.CLASS_DICT.keys())
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_df, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Anemia')
    plt.xlabel('Predicted Anemia')
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    plt.show()
    plt.close()

def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
    
def show_values(pc, fmt="%.2f", **kw):    
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    fig, ax = plt.subplots()    
    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'fontname':'Arial', 'size':'14'}
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    plt.xlim( (0, AUC.shape[1]) )

    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    plt.colorbar(c)

    show_values(c)

    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report(y_actual, y_pred, save=False, filename=False, cmap='RdBu'):
    '''
    Plots a classification report of a model's results
    '''
    cr = classification_report(y_actual, y_pred)
    lines = cr.split('\n')
    class_names = list(constants.CLASS_DICT.keys())
    plotMat = []
    support = []
    #class_names = []
    #count = 0
    for line in lines[2 : (len(lines) - 5)]:
        t = line.strip().split()
        if len(t) < 2: continue
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        plotMat.append(v)

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    ytick_labels = [f'{class_names[i]}({sup})' for i, sup in enumerate(support) ]
    
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), 'Classification Report', xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, 
            cmap=cmap)
    if save:
        plt.savefig(filename, bbox_inches = 'tight')
    plt.show()
    plt.close()


############################################################### PATHWAY VISUALISATION ################################################################

def generate_tuple_dict(df):
    frequency_dict = {}
    for traj in df.trajectory:
        if traj in frequency_dict.keys():
            frequency_dict[traj] += 1
        else:
            frequency_dict[traj] = 1
    overall_tup_dict = {}
    for key, value in frequency_dict.items():
        new_key = ast.literal_eval(key)
        for tup in zip(new_key, new_key[1:]):
            if tup in overall_tup_dict.keys():
                overall_tup_dict[tup] += value
            else:
                overall_tup_dict[tup] = value
    return overall_tup_dict

def create_sankey_df(df):
    overall_tup_dict = generate_tuple_dict(df)
    sankey_df = pd.DataFrame()
    sankey_df['Label1'] = [i[0] for i in overall_tup_dict.keys()]
    sankey_df['Label2'] = [i[1] for i in overall_tup_dict.keys()]
    sankey_df['value'] = list(overall_tup_dict.values())
    anemia_class = [i for i in sankey_df.Label2 if ((('anemia' in i) | ('Anemia' in i)) |('Inconclusive' in i))][0]
    end_row = pd.DataFrame({'Label1': anemia_class, 'Label2': '', 'value':10**-10}, index=[0])
    sankey_df = pd.concat([sankey_df.iloc[:], end_row]).reset_index(drop=True)
    # sankey_df['Label1'] = sankey_df['Label1'].replace({'Anemia of chronic disease': 'ACD', 'Iron deficiency anemia':'IDA'})
    # sankey_df['Label2'] = sankey_df['Label2'].replace({'Anemia of chronic disease': 'ACD', 'Iron deficiency anemia':'IDA'})   
    return sankey_df

def create_source_and_target(sankey_df, dmap):
    sankey_df['source'] = sankey_df['Label1'].map(dmap)
    sankey_df['target'] = sankey_df['Label2'].map(dmap)
    sankey_df.sort_values(by=['source'], inplace=True)
    return sankey_df

def draw_sankey_diagram(anems_df_list, title, save=False, filename=False):
    '''
    Draws the pathways generated by the model
    '''
    colors = ['blue', 'red', 'gray', 'yellow', 'hotpink', 'deepskyblue', 'fuchsia', 'darkturquoise']
    sankeys_df_dict = {}
    unique_actions = []
    for i, anem_df in enumerate(anems_df_list):
        sankeys_df_dict[str(i)] = create_sankey_df(anem_df)
        
    for sankey_df in sankeys_df_dict.values():
        unique_actions = unique_actions + list(sankey_df['Label1'].unique()) + list(sankey_df['Label2'].unique())
        
    unique_actions = list(set(unique_actions))
    dmap = dict(zip(unique_actions, range(len(unique_actions))))
    
    for key, sankey_df in sankeys_df_dict.items():
        samp_sankey_df = create_source_and_target(sankey_df, dmap)
        mask = samp_sankey_df['value'] >= 0.5
        samp_sankey_df['color'] = np.where(mask, colors[int(key)], 'white')
        sankeys_df_dict[key] =  samp_sankey_df
        
    nodes_color = ['green' if (('anemia' in node.lower())|('Inconclusive' in node)) else 'orange' for node in unique_actions]
    label = unique_actions
    
    target, value, source, link_color = [], [], [], []
    for sankey_df in sankeys_df_dict.values():
        target = target + list(sankey_df.target)
        value = value + list(sankey_df.value)
        source = source + list(sankey_df.source)
        link_color = link_color + list(sankey_df.color)
        
    fig = go.Figure(data=[go.Sankey(
        node = dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=label, color=nodes_color),
        link= dict(source=source, target=target, value=value, color=link_color)
    )])
    fig.update_layout(title_text=title, 
                      title_x=0.2,  
                      title_font_size=24, 
                      title_font_color='black', 
                      title_font_family='Times New Roman', 
                      font = dict(family='Times New Roman', size=18),
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      plot_bgcolor='rgba(0, 0, 0, 0)')

    if save:                 
        fig.write_image(f'../../pathways/{filename}.png')
    fig.show()


def create_sankey(df, class_list, title, save=False, filename=False):
    dfs_list = []
    for diag_class in class_list:
        dfs_list.append(df[df.y_pred==diag_class])
    
    draw_sankey_diagram(dfs_list, title, save, filename)


################################################################### OTHER ############################################################################
def generate_nans(df, column_list, frac):
    '''
    Simulating missing values in the data, frac can be a float or a list of floats same length as column_list
    '''
    for i, col in enumerate(column_list):
        if isinstance(frac, float):
            vals_to_nan = df[col].dropna().sample(frac=frac, random_state=constants.SEED).index
        elif isinstance(frac, list) & (len(column_list)==len(frac)):
            vals_to_nan = df[col].dropna().sample(frac=frac[i]).index
        elif len(column_list) != len(frac):
            print('The column and frac lists should be of the same length')
            return
        else:
            print('I have no idea what is happening :)')
            return
        df.loc[vals_to_nan, col] = np.nan
    return df


