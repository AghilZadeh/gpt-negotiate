import pandas as pd
import numpy as np
from utils import abbrev_to_us_state


df = pd.read_csv('./data/salary_data.csv')
df[['tyc', 'base', 'bonus']] *= 1000

# choosing only us jobs
states = abbrev_to_us_state.keys()
mask = df['location'].isin(states)
df_us = df[mask]

# aggregating based on selected coloumns and  calculating median and quantiles 
cols = ['company', 'level', 'title', 'location']
# creating quantile functions and rounding up to 1000
q1 = lambda a: np.round(np.percentile(a, q=25), -3)
q2 = lambda a: np.round(np.percentile(a, q=50), -3)
q3 = lambda a: np.round(np.percentile(a, q=75), -3)
df_jobs = df_us.groupby(cols)['tyc'].agg(median=q2, q1=q1, q3=q3, count='count')

# creating the output csv file with right titles
cols_2 = ['Company', 'Level', 'Title', 'Location']
df_jobs = df_jobs.reset_index().rename(columns=dict(zip(cols, cols_2)))

# saving file
df_jobs.to_csv('./data/jobs_salary.csv')
