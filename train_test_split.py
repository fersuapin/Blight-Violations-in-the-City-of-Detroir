import pandas as pd

#importing the dataset
df = pd.read_csv('dataset.csv', encoding = "ISO-8859-1", index_col = 0)

#cleaning the data from compliance "not responsible"
df.dropna(subset = ['compliance'], inplace = True)

from sklearn.model_selection import train_test_split

#Splitting into train and test sets
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

#Saving test labels for validation
df_test_labels = df_test.copy()
df_test_labels = df_test_labels[['compliance']]
df_test_labels.to_csv("test_labels.csv")

#Saving train set
df_train.to_csv("train.csv")

#formatting the test set and saving to test set csv
del df_test['payment_amount']
del df_test['payment_date']
del df_test['payment_status']
del df_test['balance_due']
del df_test['collection_status']
del df_test['compliance']
del df_test['compliance_detail']

df_test.to_csv("test.csv")
