import pandas as pd
import numpy as np
import sklearn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
random_state = 42

def prediction_to_submission(prediction, column_names, id_column):
    prediction = pd.DataFrame(prediction)
    prediction.columns  = column_names
    prediction['ImageId'] = id_column
    melted_predict = prediction.melt(id_vars='ImageId')
    id_lookup_table = pd.read_csv('Data/IdLookupTable.csv')
    melted_predict.columns = ['ImageId', 'FeatureName', 'Location']
    submission = id_lookup_table[['RowId', 'ImageId', 'FeatureName']].merge(
        melted_predict, how = 'left', on = ['ImageId', 'FeatureName'])
    submission.drop(['ImageId', 'FeatureName'],axis=1,inplace=True)
    submission['Location'] = submission.Location.round(0)
    submission['Location'][submission.Location > 96] = 96
    submission['Location'][submission.Location < 0] = 0
    submission['Location'] = submission.Location.astype(int)
    return submission