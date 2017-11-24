# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:58:39 2017

@author: TReason
"""
import os

path = "C:/Users/treason/Desktop/Personal/git/git_repo/breast_cancer_classification"

os.chdir(path)

from random_forest_classifier_project import RandomForestClassifier
import pandas as pd


df = pd.read_csv('data.csv')

rf_classifier = RandomForestClassifier(data_frame=df)

model = rf_classifier.run_pipeline(
    criteria_to_include=[True, True], 
    max_features_to_include=[True, True],
    estimators_range_to_include=range(1, 20, 5), 
    number_of_splits=10, 
    scoring_metric='roc_auc'
)


    