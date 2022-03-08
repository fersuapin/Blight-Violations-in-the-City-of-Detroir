# A $70,000,000 Machine Learning Project! Blight Violations in the City of Detroit
## Predict whether a Blight Ticket will be Paid using Machine Learning and Python.
Here you can find all the data related to my article on Medium
![image](https://user-images.githubusercontent.com/36652766/157273349-8f5f16f8-66d2-44bd-872c-3a404fa4c1c1.png)

Blight violations in the city of Detroit have been an increasingly dificult problem to solve. Every year the City of Detroit issues millions of dollars in fines to residents, many of which remain unpaid, adding up to more than $70,000,000 in unpaid fines. Enforcing these fines is extremely costly, because it forces the city to spend time and valuable resources to get it’s citizens to comply to a law that shouldn’t even exist.
Blighted properties are physical spaces or structures that exist in a state that is not only not beneficial, but detrimental to its surroundings. People who own these properties not only devalue their own assets, but others’ as well.
I will not try to get into the reasons why this problem came to be, if you would like to know more about Detroit’s decline, then please check out this article from Scott Beyer on Forbes magazine, which explains the events that led up to said decline, and why the city continues to do so poorly. We will try to focus instead on the problem at hand, helping the city enforce fines by predicting whether a person is going to comply with a property maintenance fine or not.

## Into the Data
The most up-to-date datasets for this problem can be found on the Detroit Open Data Portal, though I will be using the datasets in my GitHub profile in order to keep things consistent up to a specific date.
Throughout this project, we will be working with 3 datasets. The variables’ and files’ descriptions can also be found on my GitHub profile.

* dataset.csv
* addressess.csv
* latlons.csv

We will start our coding project by using the Pandas library for data analysis and manipulation. Importing our first dataset using pd.read_csv and dropping all rows where the individual was found “not responsible” to the blight charges (as these rows will not be useful for our analysis).
![image](https://user-images.githubusercontent.com/36652766/157273686-c294e3ef-9935-4473-ba13-5d3420281581.png)

We will then split our dataset into two diferent datasets using Scikit-learn’s train_test_split function, our training and testing set. The training set will be used to train the machine learning model, whilst the testing set will be used to evaluate the model. We will also create one last dataset containing only the labels from our test set in order to validate our model.

![image](https://user-images.githubusercontent.com/36652766/157273727-278536fb-e13d-4f19-b5b4-9ac614d4c485.png)

## Data Preprocessing
Before training our model, it is important to remove every observation that might contain information about the target (as this data will not be available when the model is actually used for prediction).
Upon inspecting our training data, we can see that the columns labeled ‘payment_amount’, ‘payment_date’, ‘payment_status’, ‘balance_due’, ‘collection_status’, and ‘compliance_detail’ might contain sensitive information that we would probably not have when using the model for prediction, so we will remove them from our dataset.
![image](https://user-images.githubusercontent.com/36652766/157273781-210291db-4ed3-40c8-ae1a-117be28d3834.png)

We will then merge our train and test datasets with our “addresses” and “latlons” datasets in order to include location data into our analysis.
![image](https://user-images.githubusercontent.com/36652766/157273810-23be96ac-93a8-4b33-a0d3-7a107f8ad686.png)

Upon further inspection, we can see that both our training and test sets now contain a lot of data that might not be actually useful for training our model. Columns like ‘admin_fee’ and ‘state_fee’ provide little to no value in order to determine if a fee will be paid (since the same value persist throughout our whole dataset), and columns like ‘mailing_address_str_name’ and ‘violator_name’ each have so many unique values that will they will provide no siginificant value and will only increase our computation time.

The final features that we will use in modeling will be the following:

‘ticket_id’, ‘agency_name’, ‘disposition’, ‘fine_amount’, ‘late_fee’,
‘discount_amount’, ‘judgment_amount’, ‘compliance’, ‘lat’, ‘lon’,
‘ticket_issued_date’, ‘violation_code’

“Side note: These specific features will probably not provide us with the best posible result, I encourage you to try out implementing/dropping other features that you might believe relevant in order to improve the model’s score” -F

## Encoding
Before training our model, we still have to tackle one last preprocessing task. Upon inspecting our remaining variables, we can see that some of our features like ‘violation_code’ or ‘agency_name’ contain categorical data that will not be useful for our machine learning model. Machine learning algorithms require that input and output variables are numbers, so we will use two diferent methods in order to transform these variables into variables that the computer can use.
Our ‘agency_name’ and ‘disposition’ columns have very little unique values (4–5 each), so we will create a dictionary and map these values to an integer and replace these values throughout our dataset.
![image](https://user-images.githubusercontent.com/36652766/157274005-b8c1e6ce-e04d-4c9d-a1b3-2aa87573fecc.png)

On the other hand, our ‘violation_code’ column, has over 150 unique values. Since creating a dictionary and mapping every single value would take us an eternity, we will use Scikit-learn’s LabelEncoder function, to create the dictionary for us and then map it into another column called ‘violation’. We will then delete the ‘violation_code’ initial column.
![image](https://user-images.githubusercontent.com/36652766/157274045-890249a7-244b-4d1c-9e98-d2ec3da885ca.png)

The correlation matrix for our remaining variables shows that ‘disposition’, ‘discount_amount’ and ‘agency_name’ are the variables more closely related to our target ‘compliance’ feature. This suggests that these variable might provide a much stronger influence than the rest of the variables when training the model.

![image](https://user-images.githubusercontent.com/36652766/157274274-75d585b9-f965-4d50-bbf0-e0e6678ca8a3.png)

Looking up specifically at the ‘discount_amount’ feature and its correlation to the target variable. It makes sense that giving out a discount to a previously issued fine might encourage an individual to comply with the City.

## Training our Model
For our project, we will use three different machine learning algorithms that we will then evaluate against our test_labels dataset. Using Scikit-learn’s Logistic Regression, Naive Bayes and Gradient Boosting classifying functions we can create our three models and fit them to our training data. We can then predict the probability that a given blight fine will be paid by applying these models to our test data. We then format our results in order to keep only the ticket_id and our prediction, so as it is consistent with our test_labels dataset.
![image](https://user-images.githubusercontent.com/36652766/157274381-71b7a49f-97a0-4eb0-91ab-cc2468665517.png)

## Evaluation
Our evaluation metric will be based on the AUC: Area Under the ROC Curve, which scores (from 0 to 1) the ability of a machine learning classifier to distinguish between True Positives and False Positives.
![image](https://user-images.githubusercontent.com/36652766/157274442-63424179-6a77-4b7a-b55f-40a54e9fb198.png)

On this example, the Gradient Boosting model outperformed the other models returning an AUC score of 0.815, while the Logistic Regression and the Naive Bayes classifiers returned scores of 0.763 and 0.709, respectively. The 0.815 AUC score from our Gradient Boosting model means 81.5% of the time, our classifier will correctly assign a higher probability of compliance to a random compliant blight ticket than to a random non-compliant ticket.

## Conclusion
This project was part of a Prediction Competition in Kaggle from the Michigan Data Science Team (MDST) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences (MSSISS) in partnership with the City of Detroit. It was introduced to me when I was taking the Applied Machine Learning in Python course from the University of Michigan on Coursera, which I HIGHLY recommend. I hope I managed to explain the process in a concise way. Pandas and Scikit-learn have been some of my most used libraries for Data Science, and I hope you will find them useful as well. Please don’t hesitate in following me on Medium, as I will be posting more articles related to my Data Science journey in the following weeks.

## About the Author
My name is Fernando Suárez, an aspiring Data Scientist from Guatemala. I like to write about basic data science concepts and play with different algorithms and Data Science tools. You could connect with me on LinkedIn.
