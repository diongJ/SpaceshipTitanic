1. Introduction

Hello Kagglers, we have a challenge here, to use our data science skills to solve the cosmic mystery and retrieve the lost passengers. This is an extensive manual on binary classification utilizing the dataset from the Spaceship Titanic.

Table of Contents:

1. INTRODUCTION
2. IMPORTS
3. EXPLORATORY DATA ANALYSIS
3.1 Target Analysis
3.2 Numerical Feature Analysis
3.2.1 Train Test Distributions
3.2.2 Feature Target Distributions
3.2.3 Bivariate Analysis
3.2.3.1 Pair Plots
3.2.3.2 Violion Plots
3.2.3.3 t-test
3.2.3.4 ANOVA
3.2.3.5 Alternate Method to Pair Plots
3.3 Categorical/Discrete Feature Analysis
3.4 Correlation Plot
4. DATA CLEANING
4.1 Passenger Group & Cabin
4.2 Name
4.3 Handling Missing Values
5. FEATURE ENGINEERING
5.1 Numerical Transformations: Transformation Selection Explained
5.2 Encoding Techniques: Multiple Encoding Techniques Implemented
5.3 TFIDF-PCA (Text Transformation)
5.4 Encoding Techniques
5.5 Group Clustered-One Hot Transformation: Something Different!
5.6 Multiplicaive Features
5.7 Less important features
5.8 Feature Elimination/Selection
6. Scaling Data
7. Model Development
7.1 Define & Tune Models : 16 Models with Hyperparameter Tuning
7.2 Model Selection
7.3 Ensembling Optimizer
7.4 Model Training
7.5 Feature Importance
7.6 Results

2. Import Libraries and Data
PassengerId	HomePlanet	CryoSleep	Cabin	Destination	Age	VIP	RoomService	FoodCourt	ShoppingMall	Spa	VRDeck	Name	Transported
0	0001_01	Europa	False	B/0/P	TRAPPIST-1e	39.0	False	0.0	0.0	0.0	0.0	0.0	Maham Ofracculy	False
1	0002_01	Earth	False	F/0/S	TRAPPIST-1e	24.0	False	109.0	9.0	25.0	549.0	44.0	Juanna Vines	True
2	0003_01	Europa	False	A/0/S	TRAPPIST-1e	58.0	True	43.0	3576.0	0.0	6715.0	49.0	Altark Susent	False
3	0003_02	Europa	False	A/0/S	TRAPPIST-1e	33.0	False	0.0	1283.0	371.0	3329.0	193.0	Solam Susent	False
4	0004_01	Earth	False	F/1/S	TRAPPIST-1e	16.0	False	303.0	70.0	151.0	565.0	2.0	Willy Santantines	True

2.1 Data Description
PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.

HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.

CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
Destination - The planet the passenger will be debarking to.
Age - The age of the passenger.
VIP - Whether the passenger has paid for special VIP service during the voyage.</font>
RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
Name - The first and last names of the passenger.
**Transported** - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

2.2 Check Missing Values
+--------------+-----------+----------------+
| Column Name  | Data Type | Non-Null Count |
+--------------+-----------+----------------+
| PassengerId  |   object  |      8693      |
|  HomePlanet  |   object  |      8492      |
|  CryoSleep   |   object  |      8476      |
|    Cabin     |   object  |      8494      |
| Destination  |   object  |      8511      |
|     Age      |  float64  |      8514      |
|     VIP      |   object  |      8490      |
| RoomService  |  float64  |      8512      |
|  FoodCourt   |  float64  |      8510      |
| ShoppingMall |  float64  |      8485      |
|     Spa      |  float64  |      8510      |
|    VRDeck    |  float64  |      8505      |
|     Name     |   object  |      8493      |
| Transported  |    bool   |      8693      |
+--------------+-----------+----------------+


Looks like we have a many missing values, let's try to deal with them with the help of EDA


3. Exploratory Data Analysis

3.1 Target Analysis

We have about same % of people who got transported and not. So, the data is indeed balanced and accuary is a good metric to choose while we build the models


3.2 Numerical Features Analysis

3.2.1 Train & Test Data Distributions

Inferences:

From the distributions of the continuous features, one thing we can clearly understand is that they are skewed and have outliers. So, we can consider options like log transformations


3.2.2 Train Data Distributions across Classes

Distributions between both the classes tell us that using these features directly into model would hinder the performance. These are the things that we could try:

Create Bins
Use algorithms that are unaffected by outliers

3.2.3 Bivariate Analysis

3.2.3.1 Pair Plots

Inferences:

The plot of Spa vs VRDeck has a good seperation between the classes. It's very clear that people who had spent less money on these were mostly Transported.
The above statement holds true for Spa vs RoomService.
VRDeck, Spa, RoomService have a good differentiation between classes.
We can create a new feature that tells the total expenditure in the above three features.

3.2.3.2 Violin Plots

Clear Confirmation from the above Violin plots that the distribution between classes are very different

3.2.3.3 t-test
The t-test is a statistical test used to determine whether the means of two groups are significantly different from each other.
The t-test produces a t-value which is used to calculate a p-value. The p-value represents the probability of observing a t-value as extreme or more extreme than the one observed if the null hypothesis (no difference between means) is true.
If the p-value is less than the chosen significance level (usually 0.05), then we reject the null hypothesis and conclude that there is a significant difference between the means. If the p-value is greater than the significance level, then we fail to reject the null hypothesis and conclude that there is not enough evidence to say that there is a significant difference between the means.
def perform_ttest(train, feature_list, target):
    """
    Performs t-test on a list of independent features for a binary classification problem
    
    :param train: pandas dataframe containing the training data
    :param feature_list: list of feature names to perform t-test on
    :param target: name of the target variable (binary)
    :return: dictionary containing t-test results
    """
    ttest_results = {}
    table = PrettyTable()

    table.field_names = ['Feature', 't_stat', 'p_val']
    
    for feature in feature_list:
        group_0 = train[train[target] == 0][feature]
        group_1 = train[train[target] == 1][feature]
        
        t_stat, p_val = ttest_ind(group_0, group_1, nan_policy='omit')
        table.add_row([feature,t_stat, p_val ])
        
    return print(table)
perform_ttest(train, cont_cols, 'Transported')
+--------------+---------------------+------------------------+
|   Feature    |        t_stat       |         p_val          |
+--------------+---------------------+------------------------+
|     Age      |  6.941461666045089  | 4.165050977515763e-12  |
| RoomService  |  23.27230572812196  | 3.400493892685092e-116 |
|  FoodCourt   |  -4.299893771259007 | 1.727865340726586e-05  |
| ShoppingMall | -0.9340564816711012 |  0.35030134449509687   |
|     Spa      |  20.914657362229097 | 9.275825095347502e-95  |
|    VRDeck    |  19.517825470012635 | 4.9897017400223456e-83 |
+--------------+---------------------+------------------------+
Inferences

All features except ShoppingMall have p-value less than 0.05, that indicates less than the significance levels and there is difference between the classes for variables.
3.2.3.4 ANOVA
from scipy.stats import f_oneway

def perform_anova(train, feature_list, target):
    """
    Performs ANOVA on a list of independent features for a binary classification problem
    
    :param train: pandas dataframe containing the training data
    :param feature_list: list of feature names to perform ANOVA on
    :param target: name of the target variable (binary)
    :return: dictionary containing ANOVA results
    """
    anova_results = {}
    table = PrettyTable()
    
    table.field_names = ['Feature', 'F-statistic', 'p-value']
    
    for feature in feature_list:
        groups = []
        for group_value in train[target].unique():
            group = train[train[target] == group_value][feature].dropna()
            groups.append(group)
        
        f_stat, p_val = f_oneway(*groups)
        table.add_row([feature, f_stat, p_val])
        
    return print(table)

perform_anova(train, cont_cols, 'Transported')
+--------------+--------------------+------------------------+
|   Feature    |    F-statistic     |        p-value         |
+--------------+--------------------+------------------------+
|     Age      | 48.18389006117339  | 4.165050977515763e-12  |
| RoomService  | 541.6002139031784  | 3.400493892685092e-116 |
|  FoodCourt   | 18.489086444112008 | 1.727865340726586e-05  |
| ShoppingMall | 0.8724615109517998 |  0.35030134449509687   |
|     Spa      | 437.4228925794437  | 9.275825095347502e-95  |
|    VRDeck    | 380.94551107787373 | 4.9897017400223456e-83 |
+--------------+--------------------+------------------------+
3.2.3.5 Alternate Method to Pair Plots
If we have many features, it would be difficult to visually plot and understand all the features. So here is a method that can tell us which pair of features together are really important in the classification task
Let's apply SVM just using a pair of two features and the target feature, see if it's able to do the the job better
Thanks @louiesagefor the suggestion

+---------------------------------+--------------------+
|           Feature Pair          |      Accuracy      |
+---------------------------------+--------------------+
|   ('RoomService', 'FoodCourt')  | 0.8149085471068676 |
|  ('FoodCourt', 'ShoppingMall')  | 0.8128379155642471 |
|      ('RoomService', 'Spa')     | 0.8076613367076958 |
|    ('RoomService', 'VRDeck')    | 0.8047854595651673 |
|     ('ShoppingMall', 'Spa')     | 0.799493845622915  |
|    ('ShoppingMall', 'VRDeck')   | 0.7968480386517888 |
|       ('FoodCourt', 'Spa')      | 0.7966179684803865 |
|        ('Spa', 'VRDeck')        | 0.7931669159093524 |
|     ('FoodCourt', 'VRDeck')     | 0.7848843897388703 |
| ('RoomService', 'ShoppingMall') | 0.7793627056252157 |
|          ('Age', 'Spa')         | 0.7378350396871046 |
|      ('Age', 'RoomService')     | 0.7302427240308293 |
|        ('Age', 'VRDeck')        | 0.7277119521454043 |
|       ('Age', 'FoodCourt')      | 0.7206948119176348 |
|     ('Age', 'ShoppingMall')     | 0.7087311630047164 |
+---------------------------------+--------------------+
Inferences:

Earlier from the pair plots, we have established that Spa, VRDeck,& RoomService are really important in the classification. Now, the above method is moore reliable than visualization because we have numbers.
FoodCourt & RoomService together has really good classification ability. Perhaps, we can try to combine all the expenditure features than just combining Spa,VRDeck, & RoomService
A note here is that it is our understanding of the data and the feature enable us to decide what combination makes a better feature. Since, all the top features in the table are expenditure features, it is easy to understand that creating a combined total expenditure would be a better option

3.3 Categorical/Discrete Analysis
['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
3.3.1 Target Distributions

Inferences

Less number of people from Earth are saved :(
Cryosleep has good difference in proportions, poeple who are in Cryosleep are more likely to be Transported
All the categories have differences in distributions in the classes
3.4 Correlation Plot

We can see that they are correlated only to a litlle extent however, our feature engineering techniques might create highly correlated features


4. Data Cleaning

4.1 Passenger Group & Cabin
Passenger Group: Since it is mentioned in the problem statement that Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
Cabin: Cabin has the format Deck/Number/Side.

4.2 Name

4.3 Handling Missing Values

4.3.1 Missing Categorical features
# Calculate the missing percentages for both train and test data
train_missing_pct = train[miss_cat].isnull().mean() * 100
test_missing_pct = test[miss_cat].isnull().mean() * 100

# Combine the missing percentages for train and test data into a single dataframe
missing_pct_df = pd.concat([train_missing_pct, test_missing_pct], axis=1, keys=['Train %', 'Test%'])

# Print the missing percentage dataframe
print(missing_pct_df)
              Train %     Test%
HomePlanet   2.312205  2.034136
Destination  2.093639  2.151040
Both train and test datasets have around the same missing values. For now, we can use some analysis and fill them. There is also an option to drop the rows with all four features missing

Iterative CatBoost Imputer: Please refer to my notebook [here](https://www.kaggle.com/code/arunklenin/ps3e15-iterative-catboost-imputer-ensemble?scriptVersionId=130271409) in which I implemented this method for the first time

train.head()
PassengerId	HomePlanet	CryoSleep	Destination	Age	VIP	RoomService	FoodCourt	ShoppingMall	Spa	VRDeck	Name	Transported	group	cabin_deck	cabin_num	cabin_side	Last_Name
0	0001_01	Europa	0.0	TRAPPIST-1e	39.0	0.0	0.0	0.0	0.0	0.0	0.0	Maham Ofracculy	0	1	B	0.0	P	ofracculy
1	0002_01	Earth	0.0	TRAPPIST-1e	24.0	0.0	109.0	9.0	25.0	549.0	44.0	Juanna Vines	1	2	F	0.0	S	vines
2	0003_01	Europa	0.0	TRAPPIST-1e	58.0	1.0	43.0	3576.0	0.0	6715.0	49.0	Altark Susent	0	3	A	0.0	S	susent
3	0003_02	Europa	0.0	TRAPPIST-1e	33.0	0.0	0.0	1283.0	371.0	3329.0	193.0	Solam Susent	0	3	A	0.0	S	susent
4	0004_01	Earth	0.0	TRAPPIST-1e	16.0	0.0	303.0	70.0	151.0	565.0	2.0	Willy Santantines	1	4	F	1.0	S	santantines
             Train %  Test%
HomePlanet       0.0    0.0
Destination      0.0    0.0

4.3.2 Missing Numerical features
['CryoSleep',
 'Age',
 'VIP',
 'RoomService',
 'FoodCourt',
 'ShoppingMall',
 'Spa',
 'VRDeck',
 'cabin_num']
               Train %     Test%
CryoSleep     2.496261  2.174421
Age           2.059128  2.127660
VIP           2.335212  2.174421
RoomService   2.082135  1.917232
FoodCourt     2.105142  2.478373
ShoppingMall  2.392730  2.291326
Spa           2.105142  2.361468
VRDeck        2.162660  1.870470
cabin_num     2.289198  2.338087
Most of the features are expenditure features and my hypothesis is that if someone is in CryoSleep, it is not possible to spend money in these activities.

# First lets fill CryoSleep, based on totdal expenditure
exp_features=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train["Expenditure"]=train[exp_features].sum(axis="columns")
test["Expenditure"]=test[exp_features].sum(axis="columns")

# Zero expenditure indicate that they are in CryoSleep
train['CryoSleep']=np.where(train['Expenditure']==0,1,0)
test['CryoSleep']=np.where(test['Expenditure']==0,1,0)

# Also, if they are VIPs, they probably would not choose to be in CryoSleep
train['VIP']=np.where(train['CryoSleep']==0,1,0)
test['VIP']=np.where(test['CryoSleep']==0,1,0)

train.drop(columns=["Expenditure"],inplace=True)
test.drop(columns=["Expenditure"],inplace=True)
for col in exp_features:
    train[col]=np.where(train["CryoSleep"]==1,0,train[col])
    test[col]=np.where(test["CryoSleep"]==1,0,test[col])    
    
# Calculate the missing percentages for both train and test data
train_missing_pct = train[miss_cont].isnull().mean() * 100
test_missing_pct = test[miss_cont].isnull().mean() * 100

# Combine the missing percentages for train and test data into a single dataframe
missing_pct_df = pd.concat([train_missing_pct, test_missing_pct], axis=1, keys=['Train %', 'Test%'])

# Print the missing percentage dataframe
print(missing_pct_df)
               Train %     Test%
CryoSleep     0.000000  0.000000
Age           2.059128  2.127660
VIP           0.000000  0.000000
RoomService   1.184861  1.239186
FoodCourt     1.173358  1.519757
ShoppingMall  1.150351  1.402852
Spa           1.288393  1.169044
VRDeck        1.173358  0.958616
cabin_num     2.289198  2.338087
Now we have filled almost many of them, the rest can be filled with KNN Imputer

miss_cont=[feature for feature in train.columns if train[feature].isnull().sum()>0 and train[feature].dtype!='O' and feature not in ['Transported']]
miss_cont
imputer=KNNImputer(n_neighbors=5)
train[miss_cont]=imputer.fit_transform(train[miss_cont])
test[miss_cont]=imputer.transform(test[miss_cont])

# Calculate the missing percentages for both train and test data
train_missing_pct = train[miss_cont].isnull().mean() * 100
test_missing_pct = test[miss_cont].isnull().mean() * 100

# Combine the missing percentages for train and test data into a single dataframe
missing_pct_df = pd.concat([train_missing_pct, test_missing_pct], axis=1, keys=['Train %', 'Test%'])

# Print the missing percentage dataframe
print(missing_pct_df)
              Train %  Test%
Age               0.0    0.0
RoomService       0.0    0.0
FoodCourt         0.0    0.0
ShoppingMall      0.0    0.0
Spa               0.0    0.0
VRDeck            0.0    0.0
cabin_num         0.0    0.0
# cb_params = {
#             'iterations': 500,
#             'depth': 6,
#             'learning_rate': 0.02,
#             'l2_leaf_reg': 0.5,
#             'random_strength': 0.2,
#             'max_bin': 150,
#             'od_wait': 80,
#             'one_hot_max_size': 70,
#             'grow_policy': 'Depthwise',
#             'bootstrap_type': 'Bayesian',
#             'od_type': 'IncToDec',
#             'eval_metric': 'RMSE',
#             'loss_function': 'RMSE',
#             'random_state': 42,
#         }
# def rmse(y1,y2):
#     return(np.sqrt(mean_squared_error(y1,y2)))

# def fill_missing_numerical(train,test,target, features, max_iterations=10):
    
#     df=pd.concat([train.drop(columns=[target,"PassengerId"]),test.drop(columns="PassengerId")],axis="rows")
#     df=df.reset_index(drop=True)
    
#     # Step 1: Store the instances with missing values in each feature
#     missing_rows = store_missing_rows(df, features)
    
#     # Step 2: Initially fill all missing values with "Missing"
#     for f in features:
#         df[f]=df[f].fillna(df[f].mean())
    
#     cat_features=[f for f in df.columns if df[f].dtype=="O"]
#     dictionary = {feature: [] for feature in features}
    
#     for iteration in tqdm(range(max_iterations), desc="Iterations"):
#         for feature in features:
#             # Skip features with no missing values
#             rows_miss = missing_rows[feature].index
            
#             missing_temp = df.loc[rows_miss].copy()
#             non_missing_temp = df.drop(index=rows_miss).copy()
#             y_pred_prev=missing_temp[feature]
#             missing_temp = missing_temp.drop(columns=[feature])
            
            
#             # Step 3: Use the remaining features to predict missing values using Random Forests
#             X_train = non_missing_temp.drop(columns=[feature])
#             y_train = non_missing_temp[[feature]]
            
#             catboost_classifier = CatBoostRegressor(**cb_params)
#             catboost_classifier.fit(X_train, y_train,cat_features=cat_features, verbose=False)
            
#             # Step 4: Predict missing values for the feature and update all N features
#             y_pred = catboost_classifier.predict(missing_temp)
#             df.loc[rows_miss, feature] = y_pred
#             error_minimize=rmse(y_pred,y_pred_prev)
#             dictionary[feature].append(error_minimize)  # Append the error_minimize value

#     for feature, values in dictionary.items():
#         iterations = range(1, len(values) + 1)  # x-axis values (iterations)
#         plt.plot(iterations, values, label=feature)  # plot the values
#         plt.xlabel('Iterations')
#         plt.ylabel('RMSE')
#         plt.title('Minimization of RMSE with iterations')
#         plt.legend()
#         plt.show()
#     train[features] = np.array(df.iloc[:train.shape[0]][features])
#     test[features] = np.array(df.iloc[train.shape[0]:][features])

#     return train,test


# train,test = fill_missing_numerical(train,test,"Transported",miss_cont,20)

5. Feature Engineering
We have already seen in EDA that VRDeck, Spa, RoomService have good distisgunshing capability. Let us combine the features to create an expenditure feature in these categories and also the total expenditure


5.1 Numerical Feature Transformations
We're going to see what transformation works better for each feature and select them, the idea is to compress the data. There could be situations where you will have to stretch the data. These are the methods applied:

**Log Transformation**: This transformation involves taking the logarithm of each data point. It is useful when the data is highly skewed and the variance increases with the mean.

                     y = log(x)
**Square Root Transformation**: This transformation involves taking the square root of each data point. It is useful when the data is highly skewed and the variance increases with the mean.

                     y = sqrt(x)
**Box-Cox Transformation**: This transformation is a family of power transformations that includes the log and square root transformations as special cases. It is useful when the data is highly skewed and the variance increases with the mean.

                     y = [(x^lambda) - 1] / lambda if lambda != 0
                     y = log(x) if lambda = 0
**Yeo-Johnson Transformation**: This transformation is similar to the Box-Cox transformation, but it can be applied to both positive and negative values. It is useful when the data is highly skewed and the variance increases with the mean.

                     y = [(|x|^lambda) - 1] / lambda if x >= 0, lambda != 0
                     y = log(|x|) if x >= 0, lambda = 0
                     y = -[(|x|^lambda) - 1] / lambda if x < 0, lambda != 2
                     y = -log(|x|) if x < 0, lambda = 2
**Power Transformation**: This transformation involves raising each data point to a power. It is useful when the data is highly skewed and the variance increases with the mean. The power can be any value, and is often determined using statistical methods such as the Box-Cox or Yeo-Johnson transformations.

                     y = [(x^lambda) - 1] / lambda if method = "box-cox" and lambda != 0
                     y = log(x) if method = "box-cox" and lambda = 0
                     y = [(x + 1)^lambda - 1] / lambda if method = "yeo-johnson" and x >= 0, lambda != 0
                     y = log(x + 1) if method = "yeo-johnson" and x >= 0, lambda = 0
                     y = [-(|x| + 1)^lambda - 1] / lambda if method = "yeo-johnson" and x < 0, lambda != 2
                     y = -log(|x| + 1) if method = "yeo-johnson" and x < 0, lambda = 2
Let's also do a grouped clustering follwed by a WOE encoding on these numerical features

+------------------+-----------------------------+---------------------+-------------------------------+
| Original Feature | Original Accuracy(CV-TRAIN) | Transformed Feature | Tranformed Accuracy(CV-TRAIN) |
+------------------+-----------------------------+---------------------+-------------------------------+
|       Age        |      0.5485979392352155     |         Age         |       0.5485979392352155      |
|   RoomService    |      0.6715703080565586     |     RoomService     |       0.6715703080565586      |
|    FoodCourt     |      0.5142101503908576     |    sqrt_FoodCourt   |       0.6177379204528921      |
|   ShoppingMall   |      0.509031519913231      |  sqrt_ShoppingMall  |       0.6311937356983188      |
|       Spa        |      0.6730687935663928     |         Spa         |       0.6730687935663928      |
|      VRDeck      |      0.6638673068529026     |        VRDeck       |       0.6638673068529026      |
|      group       |      0.5388243852757166     |        group        |       0.5388243852757166      |
|    cabin_num     |      0.5435417906696824     |      cabin_num      |       0.5435417906696824      |
|   expenditure    |      0.7775219237331852     |     expenditure     |       0.7775219237331852      |
+------------------+-----------------------------+---------------------+-------------------------------+

5.2 Categorical Features
For each categorical variable, perform the following encoding techniques:

**Count/Frequency Encoding**: Count the number of occurrences of each category and replace the category with its log count.
**Count Labeling**: Assign a label to each category based on its count, with higher counts receiving higher labels.
**WOE Binning**: Calculate the Weight of Evidence (WOE) for each category based on the target variable, where higher WOE values indicate a higher likelihood of the target variable being 1
**Target-Guided Mean Encoding**: Rank the categories based on the mean of target column across each category
**Group Clustering**: All the features created from the above mentioned encdoing techniques will be grouped and clustered followed by a Log transformation of Target-mean across clusters
**One-Hot Encoding**: Instead of applying OHE on individual features, OHE will be applied on the clusters created from all encoded features
Finally, the encoding technique will be selected based on their Accuracy CV performance on single feature model

5.3 TFIDF-PCA (Text Transformation)
Name and Last_Name has a lot of categies, so let me first handle them sing some text transformation techniques. If this doesn't work, I will drop this column

Applied **TFIDF** Text transformation creating 1000 vectors and then applied **PCA** to reduce it to 10 columns

def tf_idf(train, test, column,n,p):
    vectorizer=TfidfVectorizer(max_features=n)
    vectors_train=vectorizer.fit_transform(train[column])
    vectors_test=vectorizer.transform(test[column])
    
    svd=TruncatedSVD(p)
    x_pca_train=svd.fit_transform(vectors_train)
    x_pca_test=svd.transform(vectors_test)
    tfidf_df_train=pd.DataFrame(x_pca_train)
    tfidf_df_test=pd.DataFrame(x_pca_test)

    
    cols=[(column+"_tfidf_"+str(f)) for f in tfidf_df_train.columns]
    tfidf_df_train.columns=cols
    tfidf_df_test.columns=cols
    train=pd.concat([train,tfidf_df_train], axis="columns")
    test=pd.concat([test,tfidf_df_test], axis="columns")
    
    return (train, test)

(train,test)=tf_idf(train,test,"Last_Name",1000,5)
train.drop(columns=["Name","Last_Name"], inplace=True)
test.drop(columns=["Name","Last_Name"], inplace=True)
5.4 Encoding Techniques
cat_features=['HomePlanet', 'cabin_deck', 'Destination', 'cabin_side']
table = PrettyTable()
table.field_names = ['Feature', 'Encoded Feature', "Accuracy (CV)- Logistic regression"]

def OHE(train,test,cols,target):
    combined = pd.concat([train, test], axis=0)
    for col in cols:
        one_hot = pd.get_dummies(combined[col])
        counts = combined[col].value_counts()
        min_count_category = counts.idxmin()
        one_hot = one_hot.drop(min_count_category, axis=1)
        combined = pd.concat([combined, one_hot], axis="columns")
        combined = combined.drop(col, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # split back to train and test dataframes
    train_ohe = combined[:len(train)]
    test_ohe = combined[len(train):]
    test_ohe.reset_index(inplace=True,drop=True)
    test_ohe.drop(columns=[target],inplace=True)
    
    return train_ohe, test_ohe

for feature in cat_features:
    ## Target Guided Mean --Data Leakage Possible
    
    cat_labels=train.groupby([feature])['Transported'].mean().sort_values().index
    cat_labels2={k:i for i,k in enumerate(cat_labels,0)}
    train[feature+"_target"]=train[feature].map(cat_labels2)
    test[feature+"_target"]=test[feature].map(cat_labels2)
    
    ## Count Encoding
    
    dic=train[feature].value_counts().to_dict()
    train[feature+"_count"]=np.log1p(train[feature].map(dic))
    test[feature+"_count"]=np.log1p(test[feature].map(dic))

    
    ## Count Labeling
    
    dic2=train[feature].value_counts().to_dict()
    list1=np.arange(len(dic2.values()),0,-1) # Higher rank for high count
    # list1=np.arange(len(dic2.values())) # Higher rank for low count
    dic3=dict(zip(list(dic2.keys()),list1))
    train[feature+"_count_label"]=train[feature].replace(dic3)
    test[feature+"_count_label"]=test[feature].replace(dic3)

    
    ## WOE Binning
    cat_labels=np.log1p(train.groupby([feature])['Transported'].sum()/(train.groupby([feature])['Transported'].count()-train.groupby([feature])['Transported'].sum()))#.sort_values().index
    cat_labels2=cat_labels.to_dict()
    train[feature+"_WOE"]=train[feature].map(cat_labels2)
    test[feature+"_WOE"]=test[feature].map(cat_labels2)
    
    
    temp_cols=[feature+"_target", feature+"_count", feature+"_count_label",feature+"_WOE"]
    
    
    # It is possible to have NaN values in the test data when new categories are seen
    imputer=KNNImputer(n_neighbors=5)
    train[temp_cols]=imputer.fit_transform(train[temp_cols])
    test[temp_cols]=imputer.transform(test[temp_cols])
    
    
    if train[feature].dtype!="O":
        temp_cols.append(feature)
    else:
        train.drop(columns=[feature],inplace=True)
        test.drop(columns=[feature],inplace=True)
    # Also, doing a group clustering on all encoding types and an additional one-hot on the clusters
    
    temp_train=train[temp_cols]
    temp_test=test[temp_cols]
    
    sc=StandardScaler()
    temp_train=sc.fit_transform(temp_train)
    temp_test=sc.transform(temp_test)
    model = KMeans()


    # Initialize the KElbowVisualizer with the KMeans model and desired range of clusters
    visualizer = KElbowVisualizer(model, k=(3, 15), metric='calinski_harabasz', timings=False)

    # Fit the visualizer to the data
    visualizer.fit(np.array(temp_train))

    ideal_clusters = visualizer.elbow_value_
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Calinski-Harabasz Index')
    plt.title("Clustering on encoded featured from "+feature)
    plt.show()
    print(ideal_clusters)
    if ideal_clusters is not None:
        
        kmeans = KMeans(n_clusters=ideal_clusters)
        kmeans.fit(np.array(temp_train))
        labels_train = kmeans.labels_

        train[feature+'_cat_cluster_WOE'] = labels_train
        test[feature+'_cat_cluster_WOE'] = kmeans.predict(np.array(temp_test))

        train[feature+'_cat_OHE_cluster']=feature+"_OHE_"+train[feature+'_cat_cluster_WOE'].astype(str)
        test[feature+'_cat_OHE_cluster']=feature+"_OHE_"+test[feature+'_cat_cluster_WOE'].astype(str)

        train, test=OHE(train,test, [feature+'_cat_OHE_cluster'],"Transported")

        cat_labels=cat_labels=np.log1p(train.groupby([feature+'_cat_cluster_WOE'])['Transported'].mean())
        cat_labels2=cat_labels.to_dict()
        train[feature+'_cat_cluster_WOE']=train[feature+'_cat_cluster_WOE'].map(cat_labels2)
        test[feature+'_cat_cluster_WOE']=test[feature+'_cat_cluster_WOE'].map(cat_labels2)
        
        temp_cols=temp_cols+[feature+'_cat_cluster_WOE']
    else:
        print("No good clusters were found, skipped without clustering and OHE")
        

    
    
    
    # See which transformation along with the original is giving you the best univariate fit with target
    skf=StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    accuaries=[]
    
    for f in temp_cols:
        X=train[[f]].values
        y=train["Transported"].values
        
        acc=[]
        for train_idx, val_idx in skf.split(X,y):
            X_train,y_train=X[train_idx],y[train_idx]
            x_val,y_val=X[val_idx],y[val_idx]
            
            model=LogisticRegression()
            model.fit(X_train,y_train)
            y_pred=model.predict_proba(x_val)[:,1]
            precisions,recalls, thresholds=precision_recall_curve(y_val,y_pred)
#             cutoff=f1_cutoff(precisions,recalls, thresholds)
            cutoff=acc_cutoff(y_val,y_pred)
#             print(cutoff)
            predicted =pd.DataFrame()
            predicted["Transported"] = y_pred
            y_pred=np.where(predicted["Transported"]>float(cutoff),1,0)
            acc.append(accuracy_score(y_val,y_pred))
        accuaries.append((f,np.mean(acc)))
    best_col, best_acc=sorted(accuaries, key=lambda x:x[1], reverse=True)[0]
    
    # check correlation between best_col and other columns and drop if correlation >0.75
    corr = train[temp_cols].corr(method='pearson')
    corr_with_best_col = corr[best_col]
    cols_to_drop = [f for f in temp_cols if corr_with_best_col[f] > 0.75 and f != best_col]
    final_selection=[f for f in temp_cols if f not in cols_to_drop]
    if cols_to_drop:
        train = train.drop(columns=cols_to_drop)
        test = test.drop(columns=cols_to_drop)
    table.add_row([feature,best_col ,best_acc])
print(table)

4

3

4

None
No good clusters were found, skipped without clustering and OHE
+-------------+--------------------+------------------------------------+
|   Feature   |  Encoded Feature   | Accuracy (CV)- Logistic regression |
+-------------+--------------------+------------------------------------+
|  HomePlanet | HomePlanet_target  |         0.5866760578283932         |
|  cabin_deck | cabin_deck_target  |         0.5826493657659089         |
| Destination | Destination_target |         0.5476811766728833         |
|  cabin_side | cabin_side_target  |         0.552510217848498          |
+-------------+--------------------+------------------------------------+
No column with WOE Encoding has been selected


5.5 Clustering-One Hot Transformation
Let's take the unimportant feartures we created using transformations and use them to create clusters followed by a one hot encoding on them. We wil apply this on each subset of original features

table = PrettyTable()
table.field_names = ['Cluster WOE Feature', 'MAE(CV-TRAIN)']
for col in num_feat:
    sub_set=[f for f in unimportant_features if col in f]
    print(sub_set)
    temp_train=train[sub_set]
    temp_test=test[sub_set]
    sc=StandardScaler()
    temp_train=sc.fit_transform(temp_train)
    temp_test=sc.transform(temp_test)
    model = KMeans()


    # Initialize the KElbowVisualizer with the KMeans model and desired range of clusters
    visualizer = KElbowVisualizer(model, k=(3, 25), metric='calinski_harabasz', timings=False)

    # Fit the visualizer to the data
    visualizer.fit(np.array(temp_train))
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Calinski-Harabasz Index')
    plt.show()

    ideal_clusters = visualizer.elbow_value_
    if ideal_clusters is None:
        ideal_clusters=25

    # print(ideal_clusters)
    kmeans = KMeans(n_clusters=ideal_clusters)
    kmeans.fit(np.array(temp_train))
    labels_train = kmeans.labels_

    train[col+'_OHE_cluster'] = labels_train
    test[col+'_OHE_cluster'] = kmeans.predict(np.array(temp_test))
    # Also, making a copy to do mean encoding followed by a log transformation
    
    train[col+"_unimp_cluster_WOE"]=train[col+'_OHE_cluster']
    test[col+"_unimp_cluster_WOE"]=test[col+'_OHE_cluster'] 
    cat_labels=cat_labels=np.log1p(train.groupby([col+"_unimp_cluster_WOE"])['Transported'].mean())
    cat_labels2=cat_labels.to_dict()
    train[col+"_unimp_cluster_WOE"]=train[col+"_unimp_cluster_WOE"].map(cat_labels2)
    test[col+"_unimp_cluster_WOE"]=test[col+"_unimp_cluster_WOE"].map(cat_labels2)

    X=train[[col+"_unimp_cluster_WOE"]].values
    y=train["Transported"].values

    ACC=[]
    for train_idx, val_idx in kf.split(X,y):
        X_train,y_train=X[train_idx],y[train_idx]
        x_val,y_val=X[val_idx],y[val_idx]

        model=LogisticRegression()
        model.fit(X_train,y_train)
        y_pred=model.predict_proba(x_val)[:,1]
        precisions,recalls, thresholds=precision_recall_curve(y_val,y_pred)
#             cutoff=f1_cutoff(precisions,recalls, thresholds)
        cutoff=acc_cutoff(y_val,y_pred)
#             print(cutoff)
        predicted =pd.DataFrame()
        predicted["Transported"] = y_pred
        y_pred=np.where(predicted["Transported"]>float(cutoff),1,0)
        ACC.append(accuracy_score(y_val,y_pred))
    table.add_row([col+"_unimp_cluster_WOE",np.mean(ACC)])
    
    train[col+'_OHE_cluster']=col+"_OHE_"+train[col+'_OHE_cluster'].astype(str)
    test[col+'_OHE_cluster']=col+"_OHE_"+test[col+'_OHE_cluster'].astype(str)
    train, test=OHE(train,test,[col+'_OHE_cluster'],"Transported")
print(table)
['log_Age', 'sqrt_Age', 'bx_cx_Age', 'y_J_Age', 'pow_Age', 'pow2_Age', 'log_pow2Age', 'Age_pca_comb']

['log_RoomService', 'sqrt_RoomService', 'bx_cx_RoomService', 'y_J_RoomService', 'pow_RoomService', 'pow2_RoomService', 'log_pow2RoomService', 'RoomService_pca_comb']

['FoodCourt', 'log_FoodCourt', 'bx_cx_FoodCourt', 'y_J_FoodCourt', 'pow_FoodCourt', 'pow2_FoodCourt', 'log_pow2FoodCourt', 'FoodCourt_pca_comb']

['ShoppingMall', 'log_ShoppingMall', 'bx_cx_ShoppingMall', 'y_J_ShoppingMall', 'pow_ShoppingMall', 'pow2_ShoppingMall', 'log_pow2ShoppingMall', 'ShoppingMall_pca_comb']

['log_Spa', 'sqrt_Spa', 'bx_cx_Spa', 'y_J_Spa', 'pow_Spa', 'pow2_Spa', 'log_pow2Spa', 'Spa_pca_comb']

['log_VRDeck', 'sqrt_VRDeck', 'bx_cx_VRDeck', 'y_J_VRDeck', 'pow_VRDeck', 'pow2_VRDeck', 'log_pow2VRDeck', 'VRDeck_pca_comb']

['log_group', 'sqrt_group', 'bx_cx_group', 'y_J_group', 'pow_group', 'pow2_group', 'log_pow2group', 'group_pca_comb']

['log_cabin_num', 'sqrt_cabin_num', 'bx_cx_cabin_num', 'y_J_cabin_num', 'pow_cabin_num', 'pow2_cabin_num', 'log_pow2cabin_num', 'cabin_num_pca_comb']

['log_expenditure', 'sqrt_expenditure', 'bx_cx_expenditure', 'y_J_expenditure', 'pow_expenditure', 'pow2_expenditure', 'log_pow2expenditure', 'expenditure_pca_comb']

+--------------------------------+--------------------+
|      Cluster WOE Feature       |   MAE(CV-TRAIN)    |
+--------------------------------+--------------------+
|     Age_unimp_cluster_WOE      | 0.541466079388384  |
| RoomService_unimp_cluster_WOE  | 0.6712250836607012 |
|  FoodCourt_unimp_cluster_WOE   | 0.6103778950570744 |
| ShoppingMall_unimp_cluster_WOE | 0.6192296601986694 |
|     Spa_unimp_cluster_WOE      | 0.671688292792614  |
|    VRDeck_unimp_cluster_WOE    | 0.6635222147269288 |
|    group_unimp_cluster_WOE     | 0.5229505442905704 |
|  cabin_num_unimp_cluster_WOE   | 0.5273206089705436 |
| expenditure_unimp_cluster_WOE  | 0.758655741174292  |
+--------------------------------+--------------------+
5.6 Multiplicative Features
In this section, a new product feature if created on by multiplying all continuous original features. The final selection of features depend on the Accuracy values with a cutoff

# from itertools import combinations
# # num_features=[f for f in train.columns if train[f].nunique()>100 and f not in ['Transported',"PassengerId"]]
# feature_pairs = list(combinations(num_feat, 2))

# table = PrettyTable()
# table.field_names = ['Pair Features', 'Accuracy(CV-TRAIN)', "Selected"]


# selected_features=[]
# max_product=float('-inf')
# for pair in feature_pairs:
#     col1, col2 = pair
# #     print(pair)
#     product_col_train = train[col1] * train[col2]
#     product_col_test= test[col1] * test[col2]
#     name=f'{col1}_{col2}_product'
#     train[name] = product_col_train
#     test[name] = product_col_test
#     max_product = max(max_product, product_col_train.max())

#     kf=KFold(n_splits=5, shuffle=True, random_state=42)
#     MAE=[]
#     X=train[[name]].values
#     y=train["Transported"].values

#     ACC=[]
#     for train_idx, val_idx in kf.split(X,y):
#         X_train,y_train=X[train_idx],y[train_idx]
#         x_val,y_val=X[val_idx],y[val_idx]

#         model=LogisticRegression()
#         model.fit(X_train,y_train)
#         y_pred=model.predict_proba(x_val)[:,1]
#         precisions,recalls, thresholds=precision_recall_curve(y_val,y_pred)
# #             cutoff=f1_cutoff(precisions,recalls, thresholds)
#         cutoff=acc_cutoff(y_val,y_pred)
# #             print(cutoff)
#         predicted =pd.DataFrame()
#         predicted["Transported"] = y_pred
#         y_pred=np.where(predicted["Transported"]>float(cutoff),1,0)
#         ACC.append(accuracy_score(y_val,y_pred))
#     if np.mean(ACC)<0.7:
#         unimportant_features.append(name)
#         selected="No"
#     else:
#         selected_features.append(pair)
#         selected="Yes"
#     table.add_row([pair,np.mean(ACC),selected ])
# table.sortby = 'Accuracy(CV-TRAIN)'
# table.reversesort = True
# print(table)
5.7 Less Important Features
There are a lot of features created and many of them are not important/highly correlated, the first level of reduction is to create subsets based on the original features, apply PCA to select PC1 and drop the subset

Number of Unimportant Features are  72
test.reset_index(inplace=True,drop=True)
for col in cont_cols:
    sub_set=[f for f in unimportant_features if col in f]
    
    existing=[f for f in train.columns if f in sub_set]
    temp_train=train[existing]
    temp_test=test[existing]
    sc=StandardScaler()
    temp_train=sc.fit_transform(temp_train)
    temp_test=sc.transform(temp_test)
    
    pca=TruncatedSVD(n_components=1)
    x_pca_train=pca.fit_transform(temp_train)
    x_pca_test=pca.transform(temp_test)
    x_pca_train=pd.DataFrame(x_pca_train, columns=[col+"_pca_comb_unimp"])
    x_pca_test=pd.DataFrame(x_pca_test, columns=[col+"_pca_comb_unimp"])
    
    train=pd.concat([train,x_pca_train],axis='columns')
    test=pd.concat([test,x_pca_test],axis='columns')
    for f in sub_set:
        if f in train.columns and f not in cont_cols:
            train=train.drop(columns=[f])
            test=test.drop(columns=[f])
5.8 Feature Selection
We have create a lot of columns from transformations, clustering, encoding, PCA. Let's look at the correlation between all the features derived fro the initial numerical features


Not so much correlation but there are red/green spots, so let's reduce them Steps to Eliminate Correlated Fruit Features:

Group features based on their parent feature. For example, all features derived from Age come under one set
Apply PCA on the set, Cluster-Target Encoding on the set
See the performance of each feature on a cross-validated single feature-target model
Select the feature with highest CV-MAE
final_drop_list=[]

table = PrettyTable()
table.field_names = ['Original', 'Final Transformed feature', "Accuray(CV)- Logistic Regression"]

threshold=0.8
# It is possible that multiple parent features share same child features, so storing selected features to avoid selecting the same feature again
best_cols=[]

for col in num_feat:
    sub_set=[f for f in num_derived_list if col in f]
    # print(sub_set)
    
    correlated_features = []

    # Loop through each feature
    for i, feature in enumerate(sub_set):
        # Check correlation with all remaining features
        for j in range(i+1, len(sub_set)):
            correlation = np.abs(train[feature].corr(train[sub_set[j]]))
            # If correlation is greater than threshold, add to list of highly correlated features
            if correlation > threshold:
                correlated_features.append(sub_set[j])

    # Remove duplicate features from the list
    correlated_features = list(set(correlated_features))
    if len(correlated_features)>1:

        temp_train=train[correlated_features]
        temp_test=test[correlated_features]
        #Scale before applying PCA
        sc=StandardScaler()
        temp_train=sc.fit_transform(temp_train)
        temp_test=sc.transform(temp_test)

        # Initiate PCA
        pca=TruncatedSVD(n_components=1)
        x_pca_train=pca.fit_transform(temp_train)
        x_pca_test=pca.transform(temp_test)
        x_pca_train=pd.DataFrame(x_pca_train, columns=[col+"_pca_comb_final"])
        x_pca_test=pd.DataFrame(x_pca_test, columns=[col+"_pca_comb_final"])
        train=pd.concat([train,x_pca_train],axis='columns')
        test=pd.concat([test,x_pca_test],axis='columns')

        # Clustering
        model = KMeans()


        # Initialize the KElbowVisualizer with the KMeans model and desired range of clusters
        visualizer = KElbowVisualizer(model, k=(10, 25), metric='calinski_harabasz', timings=False)

        # Fit the visualizer to the data
        visualizer.fit(np.array(temp_train))
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Calinski-Harabasz Index')
        plt.title("Clustering on features from "+col)
        plt.show()

        ideal_clusters = visualizer.elbow_value_
        
        if ideal_clusters is None:
            ideal_clusters=10

        # print(ideal_clusters)
        kmeans = KMeans(n_clusters=ideal_clusters)
        kmeans.fit(np.array(temp_train))
        labels_train = kmeans.labels_

        train[col+'_final_cluster'] = labels_train
        test[col+'_final_cluster'] = kmeans.predict(np.array(temp_test))

        cat_labels=cat_labels=np.log1p(train.groupby([col+"_final_cluster"])['Transported'].mean())
        cat_labels2=cat_labels.to_dict()
        train[col+"_final_cluster"]=train[col+"_final_cluster"].map(cat_labels2)
        test[col+"_final_cluster"]=test[col+"_final_cluster"].map(cat_labels2)

        correlated_features=correlated_features+[col+"_pca_comb_final",col+"_final_cluster"]
        # See which transformation along with the original is giving you the best univariate fit with target
        kf=KFold(n_splits=5, shuffle=True, random_state=42)

        ACC=[]

        for f in correlated_features:
            X=train[[f]].values
            y=train["Transported"].values

            acc=[]
            for train_idx, val_idx in kf.split(X,y):
                X_train,y_train=X[train_idx],y[train_idx]
                x_val,y_val=X[val_idx],y[val_idx]

                model=LogisticRegression()
                model.fit(X_train,y_train)
                y_pred=model.predict_proba(x_val)[:,1]
                precisions,recalls, thresholds=precision_recall_curve(y_val,y_pred)
                cutoff=acc_cutoff(y_val,y_pred)
                predicted =pd.DataFrame()
                predicted["Transported"] = y_pred
                y_pred=np.where(predicted["Transported"]>float(cutoff),1,0)
                acc.append(accuracy_score(y_val,y_pred))

            if f not in best_cols:
                ACC.append((f,np.mean(acc)))
        best_col, best_acc=sorted(ACC, key=lambda x:x[1], reverse=True)[0]
        best_cols.append(best_col)

        cols_to_drop = [f for f in correlated_features if  f not in  best_cols]
        if cols_to_drop:
            final_drop_list=final_drop_list+cols_to_drop
        table.add_row([col,best_col ,best_acc])
    else:
        print(f"All features for {col} have correlation less than threshold")
        table.add_row([col,"All features selected" ,"--"])
print(table)      
All features for Age have correlation less than threshold








+--------------+----------------------------+----------------------------------+
|   Original   | Final Transformed feature  | Accuray(CV)- Logistic Regression |
+--------------+----------------------------+----------------------------------+
|     Age      |   All features selected    |                --                |
| RoomService  | RoomService_pca_comb_unimp |        0.6713460442789826        |
|  FoodCourt   |  FoodCourt_pca_comb_unimp  |        0.6177399150736075        |
| ShoppingMall | ShoppingMall_final_cluster |        0.640975694005589         |
|     Spa      |   Spa_unimp_cluster_WOE    |        0.6716899452153964        |
|    VRDeck    |  VRDeck_unimp_cluster_WOE  |        0.6635235387187985        |
|    group     |    group_final_cluster     |        0.5422740077197389        |
|  cabin_num   |  cabin_num_final_cluster   |        0.557112304136274         |
| expenditure  |        expenditure         |        0.7749928367757617        |
+--------------+----------------------------+----------------------------------+
final_drop_list=[f for f in final_drop_list if f not in cont_cols]
train.drop(columns=[*set(final_drop_list)],inplace=True)
test.drop(columns=[*set(final_drop_list)],inplace=True)

6. Scaling the Data
ID=test[['PassengerId']]
train.drop(columns=['PassengerId'],inplace=True)
test.drop(columns=['PassengerId'],inplace=True)
X_train=train.drop(['Transported'],axis=1)
y_train=train['Transported']

X_test=test.copy()
print(X_train.shape,X_test.shape)
(8693, 67) (4277, 67)

7. Model Development

7.1 Define and Tune Models
7.1.1 ANNs
If you want to apply ANNs, you can uncomment the below section. But since we had many outliers and in consideration of the training time, I would prefer tree based/other ML models.

# !pip install tensorflow
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout
sgd=tensorflow.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=True)
rms = tensorflow.keras.optimizers.RMSprop()
nadam=tensorflow.keras.optimizers.Nadam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam"
)
lrelu = lambda x: tensorflow.keras.activations.relu(x, alpha=0.1)
ann = Sequential()
ann.add(Dense(64, input_dim=X_train.shape[1], kernel_initializer='he_uniform', activation=lrelu))
ann.add(Dropout(0.1))
ann.add(Dense(16,  kernel_initializer='he_uniform', activation=lrelu))
ann.add(Dropout(0.1))
# model.add(Dense(32,  kernel_initializer='he_uniform', activation='relu'))
# model.add(Dropout(0.1))

ann.add(Dense(1,  kernel_initializer='he_uniform', activation='sigmoid'))
ann.compile(loss="binary_crossentropy", optimizer=nadam,metrics=['accuracy'])
7.1.2 XGBoost Tuning
# A=time.time()
# # Set up the XGBoost classifier with default hyperparameters
# xgb_params = {
#     'n_estimators': 500,
#     'learning_rate': 0.05,
#     'max_depth': 7,
#     'subsample': 1.0,
#     'colsample_bytree': 1.0,
#     'n_jobs': -1,
#     'eval_metric': 'logloss',
#     'objective': 'binary:logistic',
#     'verbosity': 0,
#     'random_state': 1,
# }
# model = xgb.XGBClassifier(**xgb_params)

# # Define the hyperparameters to tune and their search ranges
# param_dist = {
#     'n_estimators': np.arange(50, 1000,50),
#     'max_depth': np.arange(3, 15,2),
#     'learning_rate': np.arange(0.001, 0.05,0.004),
#     'subsample': [0.1,0.3,0.5,0.7,0.9],
#     'colsample_bytree': [0.1,0.3,0.5,0.7,0.9],
# }

# # Set up the RandomizedSearchCV object with cross-validation
# random_search = RandomizedSearchCV(model, param_distributions=param_dist, cv=3, n_iter=50, random_state=1, n_jobs=-1)

# # Fit the RandomizedSearchCV object to the training data
# random_search.fit(X_train, y_train)

# # Print the best hyperparameters and corresponding mean cross-validation score
# print("Best hyperparameters: ", random_search.best_params_)
# print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

# # Evaluate the best model on the test data
# best_model = random_search.best_estimator_
# print(best_model)

# xgb_params=random_search.best_params_

# B=time.time()
# print((B-A)/60) 
xgb_params={'colsample_bytree': 0.8498791800104656, 'learning_rate': 0.020233442882782587, 'max_depth': 4, 'n_estimators': 469, 'subsample': 0.746529796772373}
7.1.3 LightGBM Tuning
# # Set up the LightGBM classifier with default hyperparameters
# lgb_params = {
#     'n_estimators': 100,
#     'max_depth': 7,
#     'learning_rate': 0.05,
#     'subsample': 0.2,
#     'colsample_bytree': 0.56,
#     'reg_alpha': 0.25,
#     'reg_lambda': 5e-08,
#     'objective': 'binary',
#     'metric': 'accuracy',
#     'boosting_type': 'gbdt',
#     'device': 'cpu',
#     'random_state': 1,
# }
# model = lgb.LGBMClassifier(**lgb_params)

# # Define the hyperparameters to tune and their search ranges
# param_dist = {
#     'n_estimators': np.arange(50, 1000,50),
#     'max_depth': np.arange(3, 15,2),
#     'learning_rate': np.arange(0.001, 0.02,0.002),
#     'subsample': [0.1,0.3,0.5,0.7,0.9],
#     'colsample_bytree': [0.1,0.3,0.5,0.7,0.9],
#     'reg_alpha': [uniform(0, 1),uniform(0, 1),uniform(0, 1),uniform(0, 1)],
#     'reg_lambda': [uniform(0, 1),uniform(0, 1),uniform(0, 1),uniform(0, 1)],
# }

# # Set up the RandomizedSearchCV object with cross-validation
# random_search = RandomizedSearchCV(model, param_distributions=param_dist, cv=3, n_iter=20, random_state=1, n_jobs=-1)

# # Fit the RandomizedSearchCV object to the training data
# random_search.fit(X_train, y_train)

# # Print the best hyperparameters and corresponding mean cross-validation score
# print("Best hyperparameters: ", random_search.best_params_)
# print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

# # Evaluate the best model on the test data
# best_model = random_search.best_estimator_

# lgb_params=random_search.best_params_
lgb_params={'colsample_bytree': 0.7774799983649324, 'learning_rate': 0.007653648135411494, 'max_depth': 5, 'n_estimators': 350, 'reg_alpha': 0.14326300616140863, 'reg_lambda': 0.9310129332502252, 'subsample': 0.6189257947519665}
7.1.4 CatBoost Tuning
# # define the hyperparameter search space
# param_distributions = {
#     'depth':  np.arange(3, 15,2),
#     'learning_rate': np.arange(0.001, 0.02,0.002),
#     'l2_leaf_reg': [0.1, 0.5, 0.7],
#     'random_strength': [0.1, 0.2, 0.3],
#     'max_bin': [100, 150, 200],
#     'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
#     'bootstrap_type': ['Bayesian', 'Bernoulli'],
#     'one_hot_max_size': [10, 50, 70],
# }

# # create a CatBoostClassifier model with default parameters
# model = CatBoostClassifier(iterations=200, eval_metric='Accuracy', loss_function='Logloss', task_type='CPU')

# # perform random search with cross-validation
# random_search = RandomizedSearchCV(
#     estimator=model,
#     param_distributions=param_distributions,
#     n_iter=50,  # number of parameter settings that are sampled
#     scoring='neg_log_loss',  # use negative log-loss as the evaluation metric
#     cv=3,  # 5-fold cross-validation
#     verbose=1,
#     random_state=42
# )

# # fit the random search object to the training data
# random_search.fit(X_train, y_train)

# # print the best parameters and best score
# print('Best score:', -1 * random_search.best_score_)
# print('Best parameters:', random_search.best_params_)

# cat_params=random_search.best_params_
cat_params={'random_strength': 0.1, 'one_hot_max_size': 10, 'max_bin': 100, 'learning_rate': 0.01, 'l2_leaf_reg': 0.5, 'grow_policy': 'Lossguide', 'depth': 5, 'bootstrap_type': 'Bernoulli'}
7.1.5 Logistic Tuning
# from sklearn.model_selection import GridSearchCV

# # define the hyperparameter search space
# param_grid = {
#     'penalty': ['l1', 'l2', 'elasticnet'],
#     'C': [0.001,0.01, 0.1, 1, 10, 100],
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# }

# # create a LogisticRegression model with default parameters
# model = LogisticRegression(max_iter=500, random_state=2023)

# # perform grid search with cross-validation
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     scoring='roc_auc',  # use accuracy as the evaluation metric
#     cv=5,  # 5-fold cross-validation
#     verbose=1,
#     n_jobs=-1
# )

# # fit the grid search object to the training data
# grid_search.fit(X_train, y_train)

# # print the best parameters and best score
# print('Best score:', grid_search.best_score_)
# print('Best parameters:', grid_search.best_params_)
# lg_params=grid_search.best_params_
7.1.6 Random Forests Tuning
7.1.7 HistGBM Tuning
7.1.8 GBM Tuning
# # Define the hyperparameter search space
# param_dist = {
#     'n_estimators': np.arange(100, 1000, 50),
#     'learning_rate': np.logspace(-4, 0, num=100),
#     'max_depth': [2, 3, 4, 5, 6],
#     'min_samples_split': [2, 3, 4, 5, 6],
#     'min_samples_leaf': [1, 2, 3, 4, 5],
#     'max_features': ['sqrt', 'log2', None]
# }

# # Create the GradientBoostingClassifier model
# model = GradientBoostingClassifier(max_depth=4, max_features='sqrt',
#                                    min_samples_leaf=2, min_samples_split=5,
#                                    n_estimators=341, random_state=42)

# # Create the random search object
# random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100,
#                                    cv=5, scoring='accuracy', n_jobs=-1, random_state=42)

# # Fit the random search object to the data
# random_search.fit(X_train, y_train)
# best_model = random_search.best_estimator_

# # Print the best parameters and best score
# print("Best parameters: ", random_search.best_params_)
# print("Best score: ", random_search.best_score_)

# gbm_params=random_search.best_params_
7.1.9 SVM Tuning
7.1.10 KNN Tuning
7.1.11 MLP Tuning
7.1.12 GPC Tuning
7.1.13 ExtraTrees Tuning
7.1.14 DecisionTrees Tuning
7.1.15 AdaBoost Tuning
7.1.16 Naive Bayes Tuning

7.2 Model Selection
Kudos to tetsu2131( [http://www.kaggle.com/tetsutani]()) for this framework, the below parts of the code has been taken and modified from. Please support the account if you this work.

class Splitter:
    def __init__(self, kfold=True, n_splits=5):
        self.n_splits = n_splits
        self.kfold = kfold

    def split_data(self, X, y, random_state_list):
        if self.kfold:
            for random_state in random_state_list:
                kf = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        else:
            X_train, X_val = X.iloc[:int(X_train.shape[0]/10)], X.iloc[int(X_train.shape[0]/10):]
            y_train, y_val = y.iloc[:int(X_train.shape[0]/10)], y.iloc[int(X_train.shape[0]/10):]
            yield X_train, X_val, y_train, y_val

class Classifier:
    def __init__(self, n_estimators=100, device="cpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.len_models = len(self.models)
        
    def _define_model(self):
        xgb_params.update({
            'n_estimators': self.n_estimators,
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'random_state': self.random_state,
        })
        if self.device == 'gpu':
            xgb_params.update({
            'tree_method' :'gpu_hist',
            'predictor': 'gpu_predictor',
          })

        lgb_params.update({
            'n_estimators': self.n_estimators,
            'objective': 'binary',
            'random_state': self.random_state,
        })

        cat_params.update({
            'n_estimators': self.n_estimators,
            'task_type': self.device.upper(),
            'random_state': self.random_state,
        })
        
        cat_sym_params = cat_params.copy()
        cat_sym_params['grow_policy'] = 'SymmetricTree'
        cat_dep_params = cat_params.copy()
        cat_dep_params['grow_policy'] = 'Depthwise'
        dt_params= {'min_samples_split': 80, 'min_samples_leaf': 30, 'max_depth': 8, 'criterion': 'gini'}
#         rf_params.update({
#             'n_estimators': self.n_estimators,
#         })
        models = {
            'xgb': xgb.XGBClassifier(**xgb_params),
            'lgb': lgb.LGBMClassifier(**lgb_params),
            'cat': CatBoostClassifier(**cat_params),
#             "cat_sym": CatBoostClassifier(**cat_sym_params),
#             "cat_dep": CatBoostClassifier(**cat_dep_params),
#             'lr': LogisticRegression(),
#             'rf': RandomForestClassifier(max_depth= 9,max_features= 'auto',min_samples_split= 10,
#                                                            min_samples_leaf= 4,  n_estimators=500,random_state=self.random_state),
#             'hgb': HistGradientBoostingClassifier(max_iter=self.n_estimators,learning_rate=0.01, loss="binary_crossentropy", 
#                                                   n_iter_no_change=300,random_state=self.random_state),
#             'gbdt': GradientBoostingClassifier(**gbm_params,random_state=self.random_state),
#             'svc': SVC(gamma="auto", probability=True),
#             'knn': KNeighborsClassifier(n_neighbors=5),
#             'mlp': MLPClassifier(random_state=self.random_state, max_iter=1000),
#             'gpc': GaussianProcessClassifier(**gpc_params, random_state=self.random_state),
#             'etr':ExtraTreesClassifier(min_samples_split=55, min_samples_leaf= 15, max_depth=10,
#                                        n_estimators=200,random_state=self.random_state),
#             'dt' :DecisionTreeClassifier(**dt_params,random_state=self.random_state),
#             'ada': AdaBoostClassifier(random_state=self.random_state),
#             'GNB': GaussianNB(**nb_params),
#             'ann':ann,
        }
        
        return models

7.3 Optimizer
class OptunaWeights:
    def __init__(self, random_state):
        self.study = None
        self.weights = None
        self.random_state = random_state

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 0, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        # Calculate the Recall score for the weighted prediction
        precisions,recalls, thresholds=precision_recall_curve(y_true,weighted_pred)
#         cutoff=f1_cutoff(precisions,recalls, thresholds)
        cutoff=acc_cutoff(y_true,weighted_pred)
        y_weight_pred=np.where(weighted_pred>float(cutoff),1,0)        
        score = metrics.accuracy_score(y_true, y_weight_pred)
        return score

    def fit(self, y_true, y_preds, n_trials=2000):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        self.study = optuna.create_study(sampler=sampler, study_name="OptunaWeights", direction='maximize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds, n_trials=2000):
        self.fit(y_true, y_preds, n_trials=n_trials)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights
    
def acc_cutoff_class(y_valid, y_pred_valid):
    y_valid=np.array(y_valid)
    y_pred_valid=np.array(y_pred_valid)
    fpr, tpr, threshold = metrics.roc_curve(y_valid, y_pred_valid)
    pred_valid = pd.DataFrame({'label': y_pred_valid})
    thresholds = np.array(threshold)
    pred_labels = (pred_valid['label'].values > thresholds[:, None]).astype(int)
    acc_scores = (pred_labels == y_valid).mean(axis=1)
    acc_df = pd.DataFrame({'threshold': threshold, 'test_acc': acc_scores})
    acc_df.sort_values(by='test_acc', ascending=False, inplace=True)
    cutoff = acc_df.iloc[0, 0]
    y_pred_valid=np.where(y_pred_valid<float(cutoff),0,1)
    return y_pred_valid

7.4 Model Training
kfold = True
n_splits = 1 if not kfold else 10
random_state = 2023
random_state_list = [2140] # used by split_data [71]
n_estimators = 9999 # 9999
early_stopping_rounds = 200
verbose = False
device = 'cpu'

splitter = Splitter(kfold=kfold, n_splits=n_splits)

# Initialize an array for storing test predictions
test_predss = np.zeros(X_test.shape[0])
ensemble_score = []
weights = []
trained_models = {'xgb':[], 'lgb':[], 'cat':[]}

    
for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
    n = i % n_splits
    m = i // n_splits
            
    # Get a set of Regressor models
    classifier = Classifier(n_estimators, device, random_state)
    models = classifier.models
    
    # Initialize lists to store oof and test predictions for each base model
    oof_preds = []
    test_preds = []
    
    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions
    for name, model in models.items():
        if ('cat' in name) or ("lgb" in name) or ("xgb" in name):
            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        elif name in 'ann':
            model.fit(X_train_, y_train_, validation_data=(X_val, y_val),batch_size=5, epochs=50,verbose=verbose)
        else:
            model.fit(X_train_, y_train_)
        
        if name in 'ann':
            test_pred = np.array(model.predict(X_test))[:, 0]
            y_val_pred = np.array(model.predict(X_val))[:, 0]
        else:
            test_pred = model.predict_proba(X_test)[:, 1]
            y_val_pred = model.predict_proba(X_val)[:, 1]

#         score = roc_auc_score(y_val, y_val_pred)
        score = accuracy_score(y_val, acc_cutoff_class(y_val, y_val_pred))

        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] Accuracy score: {score:.5f}')
        
        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)
        
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))
    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(random_state=random_state)
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
    
#     score = roc_auc_score(y_val, y_val_pred)
    score = accuracy_score(y_val, acc_cutoff_class(y_val, y_val_pred))
    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] Accuracy score {score:.5f}')
    ensemble_score.append(score)
    weights.append(optweights.weights)
    
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))
    
    gc.collect()
xgb [FOLD-0 SEED-2140] Accuracy score: 0.80460
lgb [FOLD-0 SEED-2140] Accuracy score: 0.80345
cat [FOLD-0 SEED-2140] Accuracy score: 0.80920
Ensemble [FOLD-0 SEED-2140] Accuracy score 0.80920
xgb [FOLD-1 SEED-2140] Accuracy score: 0.80345
lgb [FOLD-1 SEED-2140] Accuracy score: 0.79885
cat [FOLD-1 SEED-2140] Accuracy score: 0.80345
Ensemble [FOLD-1 SEED-2140] Accuracy score 0.80805
xgb [FOLD-2 SEED-2140] Accuracy score: 0.81839
lgb [FOLD-2 SEED-2140] Accuracy score: 0.82184
cat [FOLD-2 SEED-2140] Accuracy score: 0.81839
Ensemble [FOLD-2 SEED-2140] Accuracy score 0.82529
xgb [FOLD-3 SEED-2140] Accuracy score: 0.82048
lgb [FOLD-3 SEED-2140] Accuracy score: 0.82278
cat [FOLD-3 SEED-2140] Accuracy score: 0.82854
Ensemble [FOLD-3 SEED-2140] Accuracy score 0.82509
xgb [FOLD-4 SEED-2140] Accuracy score: 0.83084
lgb [FOLD-4 SEED-2140] Accuracy score: 0.83659
cat [FOLD-4 SEED-2140] Accuracy score: 0.83084
Ensemble [FOLD-4 SEED-2140] Accuracy score 0.83429
xgb [FOLD-5 SEED-2140] Accuracy score: 0.83314
lgb [FOLD-5 SEED-2140] Accuracy score: 0.83544
cat [FOLD-5 SEED-2140] Accuracy score: 0.83199
Ensemble [FOLD-5 SEED-2140] Accuracy score 0.83544
xgb [FOLD-6 SEED-2140] Accuracy score: 0.82854
lgb [FOLD-6 SEED-2140] Accuracy score: 0.82509
cat [FOLD-6 SEED-2140] Accuracy score: 0.82509
Ensemble [FOLD-6 SEED-2140] Accuracy score 0.82969
xgb [FOLD-7 SEED-2140] Accuracy score: 0.82394
lgb [FOLD-7 SEED-2140] Accuracy score: 0.81933
cat [FOLD-7 SEED-2140] Accuracy score: 0.82048
Ensemble [FOLD-7 SEED-2140] Accuracy score 0.82509
xgb [FOLD-8 SEED-2140] Accuracy score: 0.80667
lgb [FOLD-8 SEED-2140] Accuracy score: 0.81128
cat [FOLD-8 SEED-2140] Accuracy score: 0.80898
Ensemble [FOLD-8 SEED-2140] Accuracy score 0.81128
xgb [FOLD-9 SEED-2140] Accuracy score: 0.80552
lgb [FOLD-9 SEED-2140] Accuracy score: 0.81128
cat [FOLD-9 SEED-2140] Accuracy score: 0.81128
Ensemble [FOLD-9 SEED-2140] Accuracy score 0.81703
# Calculate the mean Accuracy score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Ensemble Accuracy score {mean_score:.5f} ± {std_score:.5f}')

# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')
Ensemble Accuracy score 0.82204 ± 0.00959
--- Model Weights ---
xgb: 0.42904 ± 0.29399
lgb: 0.59984 ± 0.19804
cat: 0.42340 ± 0.22340
Based on the validation set, we can decide the decision boundary that maximizes Accuracy

precisions,recalls, thresholds=precision_recall_curve(y_val,y_val_pred)
# cutoff=f1_cutoff(precisions,recalls, thresholds)
cutoff=acc_cutoff(y_val,y_val_pred)
cutoff
0.5958020593085614
y_test_pred=np.where(test_predss>float(cutoff),1,0)

7.5 Feature importance Visualization (XGBoost, LightGBM, Catboost)
def visualize_importance(models, feature_cols, title, top=20):
    importances = []
    feature_importance = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["importance"] = model.feature_importances_
        _df["feature"] = pd.Series(feature_cols)
        _df["fold"] = i
        _df = _df.sort_values('importance', ascending=False)
        _df = _df.head(top)
        feature_importance = pd.concat([feature_importance, _df], axis=0, ignore_index=True)
        
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    # display(feature_importance.groupby(["feature"]).mean().reset_index().drop('fold', axis=1))
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance, color='skyblue', errorbar='sd')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'{title} Feature Importance [Top {top}]', fontsize=18)
    plt.grid(True, axis='x')
    plt.show()
    
for name, models in trained_models.items():
    visualize_importance(models, list(X_train.columns), name)



We can see that the transformation techniques had worked well and the column Expenditure created using bivariate analysis has been helpful. We can also see TFIDF based columns, cluster encoded columns worked well


7.6 Results
sub = pd.read_csv('../input/spaceship-titanic/sample_submission.csv')
sub['Transported'] = np.where(test_predss>cutoff,1,0).astype(bool)
sub.to_csv('submission_model.csv',index=False)
sub
PassengerId	Transported
0	0013_01	True
1	0018_01	False
2	0019_01	True
3	0021_01	True
4	0023_01	True
...	...	...
4272	9266_02	False
4273	9269_01	False
4274	9271_01	True
4275	9273_01	True
4276	9277_01	False
4277 rows × 2 columns

sub["Transported"].value_counts()/sub["Transported"].shape[0]
False    0.56909
True     0.43091
Name: Transported, dtype: float64
8. Experiment
We can use the results from different models and use AND/OR gates to identify data points that are difficult to predict. This would work because these are coming from possibly different feature engineering methods and ensemble models

In this case I'm selecting the publicly available top scored submissions

Notebook by @viktortaran
Notebook by @danutstinga
Notebook by @jimliu
sub1=pd.read_csv("/kaggle/input/space-titanic/XGB_best.csv")
sub2=pd.read_csv("/kaggle/input/solution/submission.csv")
sub3=pd.read_csv("/kaggle/input/0-81669-misaelcribeiro-solution-modularity-fe/submission.csv")
Apply an OR Gate to increase bads
sub_combined=sub1.copy()
sub1.Transported.value_counts()/sub1.shape[0]
True     0.53589
False    0.46411
Name: Transported, dtype: float64
sub_combined['Transported']=sub1['Transported'] | sub2['Transported'] |sub3['Transported'] |sub["Transported"]
sub_combined['Transported'].value_counts()/sub1.shape[0]
True     0.579846
False    0.420154
Name: Transported, dtype: float64
sub_combined.to_csv('submission.csv',index=False)
