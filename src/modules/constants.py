SEED = 42 #placeholder -- change this to be an argument fed in during the run.py

MAX_STEPS = 10

ACTION_SPACE = ['No anemia', 'Vitamin B12/Folate deficiency anemia', 'Unspecified anemia', 'Anemia of chronic disease', 
'Iron deficiency anemia', 'Hemolytic anemia', 'Aplastic anemia', 'Inconclusive diagnosis', 'hemoglobin', 'ferritin', 'ret_count',
'segmented_neutrophils', 'tibc', 'mcv', 'serum_iron', 'rbc', 'gender', 'creatinine', 'cholestrol', 'copper', 'ethanol', 'folate', 
'glucose', 'hematocrit', 'tsat']

ACTION_NUM = len(ACTION_SPACE)

CLASS_DICT = {'No anemia': 0, 'Vitamin B12/Folate deficiency anemia': 1, 'Unspecified anemia': 2, 'Anemia of chronic disease': 3, 
'Iron deficiency anemia': 4, 'Hemolytic anemia': 5, 'Aplastic anemia': 6, 'Inconclusive diagnosis': 7}

CLASS_NUM = len(CLASS_DICT)

FEATURE_NUM = ACTION_NUM - CLASS_NUM