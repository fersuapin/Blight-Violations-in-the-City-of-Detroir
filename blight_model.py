import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sn
import matplotlib.pyplot as plt


df_test_labels = pd.read_csv('test_labels.csv', encoding = "ISO-8859-1", index_col = 0)
labels = df_test_labels.sort_index().squeeze()

#Data Preprocessing
def data_preprocessing():
    df_train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
    df_test = pd.read_csv('test.csv', encoding = "ISO-8859-1")
    df_addresses = pd.read_csv('addresses.csv', encoding = "ISO-8859-1")
    df_latlons = pd.read_csv('latlons.csv', encoding = "ISO-8859-1")

    
    #Data Leakage
    del df_train['payment_amount']
    del df_train['payment_date']
    del df_train['payment_status']
    del df_train['balance_due']
    del df_train['collection_status']
    del df_train['compliance_detail']
    
    #merging with addresses and latlons
    df_train = df_train.merge(df_addresses, on = "ticket_id")
    df_train = df_train.merge(df_latlons, on = "address")
    df_test = df_test.merge(df_addresses, on = "ticket_id")
    df_test = df_test.merge(df_latlons, on = "address")

    #Useless Information
    del df_train['admin_fee']
    del df_train['state_fee']
    del df_test['admin_fee']
    del df_test['state_fee']
    del df_train['grafitti_status']
    del df_test['grafitti_status']    
    del df_train['clean_up_cost']
    del df_test['clean_up_cost']
    del df_train['inspector_name']
    del df_test['inspector_name']
    del df_train['hearing_date']
    del df_test['hearing_date']
    del df_train['violation_description']
    del df_test['violation_description']
    del df_test['violator_name']
    del df_train['violator_name']

    #useless info from other countries
    df_train.drop(df_train.loc[df_train['country']=='Aust'].index, inplace=True)
    df_train.drop(df_train.loc[df_train['country']=='Egyp'].index, inplace=True)
    df_train.drop(df_train.loc[df_train['country']=='Cana'].index, inplace=True)
    df_train.drop(df_train.loc[df_train['country']=='Germ'].index, inplace=True)
    
    #useless info on addresses
    del df_train['violation_street_number']
    del df_train['violation_street_name']
    del df_train['violation_zip_code']
    del df_train['mailing_address_str_number']
    del df_train['mailing_address_str_name']
    del df_train['zip_code']
    del df_train['country']
    del df_train['non_us_str_code']
    del df_train['city']
    del df_train['state']
    del df_test['violation_street_number']
    del df_test['violation_street_name']
    del df_test['violation_zip_code']
    del df_test['mailing_address_str_number']
    del df_test['mailing_address_str_name']
    del df_test['zip_code']
    del df_test['country']
    del df_test['non_us_str_code']
    del df_test['city']
    del df_test['state']
    del df_test['address']
    del df_train['address']

    #filling null values in lat and lon columns for train and test
    df_test['lat'].fillna(42, inplace = True)
    df_test['lon'].fillna(-83.5, inplace = True)
    df_train['lat'].fillna(42, inplace = True)
    df_train['lon'].fillna(-83.5, inplace = True)
    
    #creating new "month" column and deleting the ticket_issued_date
    df_train['ticket_issued_date'] =  pd.to_datetime(df_train['ticket_issued_date'], format='%Y-%m-%d %H:%M:%S')
    df_test['ticket_issued_date'] =  pd.to_datetime(df_test['ticket_issued_date'], format='%Y-%m-%d %H:%M:%S')
    df_train['month'] = df_train['ticket_issued_date'].dt.month
    df_test['month'] = df_test['ticket_issued_date'].dt.month
    del df_train['ticket_issued_date']
    del df_test['ticket_issued_date']
    
    #factorizing the violation_code column into "violation", then deleting violation_code
    enc = LabelEncoder()
    df_train['violation'] = enc.fit_transform(df_train['violation_code'])
    enc_dict = dict(zip(enc.classes_, enc.transform(enc.classes_)))

    df_test['violation'] = df_test['violation_code']
    df_test.replace({'violation':enc_dict}, inplace = True)

    counter = 0
    for i in df_test['violation'].unique():
        if type(i) != int:
            enc_dict[i] = 189+counter
            counter +=1

    df_test.replace({'violation':enc_dict}, inplace = True)
    df_test['violation'] = df_test['violation'].astype('int32')

    del df_test['violation_code']
    del df_train['violation_code']
    
    #Converting the Agency Name Variable to Numeric Encoding
    cleanup_nums = {"agency_name":     {"Buildings, Safety Engineering & Env Department": 1, 
                                        "Department of Public Works": 2, 
                                        "Health Department": 3, 
                                        "Detroit Police Department": 5, 
                                        "Neighborhood City Halls": 5}}
    df_train = df_train.replace(cleanup_nums)                
    df_test = df_test.replace(cleanup_nums)
    
    #Converting the Disposition Variable to Numeric Encoding
    cleanup_nums = {"disposition":     {"Responsible by Default": 0, 
                                        "Responsible by Admission": 1, 
                                        "Responsible by Determination": 2, 
                                        "Responsible (Fine Waived) by Deter": 3}}
    df_train = df_train.replace(cleanup_nums)    
    df_test = df_test.replace(cleanup_nums)
    
    return df_test, df_train

df_test, df_train = data_preprocessing()
corrMatrix = df_train.corr()
fig, ax = plt.subplots(figsize=(8,8))
ax.set_title('Correlation Matrix for Remaining Variables')
sn.heatmap(corrMatrix, vmin=corrMatrix.values.min(), vmax=1, square=True, cmap="YlGnBu", linewidths=0.1, annot=True, annot_kws={"fontsize":8}, fmt='.2f')  
plt.show()
print(corrMatrix['compliance'].sort_values(ascending = True))


# Creating our model
models = [LogisticRegression(max_iter = 200), 
          GradientBoostingClassifier(learning_rate = 0.05),
          GaussianNB()]
def blight_model(df_train, df_test, models):

    #training the model
    y_train = df_train['compliance']
    X_train = df_train.loc[:, df_train.columns != 'compliance']

    clf = models.fit(X_train, y_train)
    df_test['prediction'] = clf.predict_proba(df_test)[:,1]
    
    result = df_test[['ticket_id', 'prediction']].copy()
    
    result.drop(result.columns.difference(['ticket_id','prediction']), 1, inplace=True)
    result.set_index(result['ticket_id'], drop=True, inplace = True)
    del result['ticket_id']
    result = result.sort_index().squeeze()
    
    return result

#Evaluating the Classifiers using the ROC AUC Score
for i in models:
    AUC = roc_auc_score(labels, blight_model(df_train, df_test, i))
    print(AUC)
    del df_test['prediction']




