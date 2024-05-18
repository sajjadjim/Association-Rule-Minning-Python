import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import datetime
import seaborn as sns

df = pd.read_csv("bread basket.csv")
print(df.head())

# Display basic information about the dataframe
#print(df.info())

# Display the unique values in the 'date_time' column
#print(df['date_time'].unique())


# Converting the 'date_time' column into the right format
df['date_time'] = pd.to_datetime(df['date_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
#print(df.head(10))

# Count of unique customers
#print("Total Number of count at Last Posititon ->>>>",df['Transaction'].nunique())
##########################################################################################################
# Extracting date
df['date'] = df['date_time'].dt.date
#Extracting time
df['time'] = df['date_time'].dt.time
# Extracting month and replacing it with text
df['month'] = df['date_time'].dt.month
df['month'] = df['month'].replace((1,2,3,4,5,6,7,8,9,10,11,12),('January','February','March','April','May','June','July','August','September','October','November','December'))
# Extracting hour
df['hour'] = df['date_time'].dt.hour
# Replacing hours with text
hour_in_num = (1,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
hour_in_obj = ('1-2','7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15',
               '15-16','16-17','17-18','18-19','19-20','20-21','21-22','22-23','23-24')
df['hour'] = df['hour'].replace(hour_in_num, hour_in_obj)

# Extracting weekday and replacing it with text
df['weekday'] = df['date_time'].dt.weekday
df['weekday'] = df['weekday'].replace((0,1,2,3,4,5,6),('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))

# dropping date_time column
print(df.drop('date_time', axis = 1, inplace = True))
print(df.head())

############################################# Data Visualization Histogram ##########################################

plt.figure(figsize=(15,5))
sns.barplot(x = df.Item.value_counts().head(20).index, y = df.Item.value_counts().head(20).values, palette = 'gnuplot')
plt.xlabel('Items', size = 15)
plt.xticks(rotation=45)
plt.ylabel('Count of Items', size = 15)
print(plt.title('Top 20 Items purchased by customers', color = 'green', size = 20))
print(plt.show())

