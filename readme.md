# Bengaluru_House_Price_Prediction

* This is Bengaluru House Price Pridiction Project
* With the help of this project we can predict the housing price using various features like size, area, bedrooms, locality, etc.<br/>
## Link to dataset:
(https://github.com/lakshayd760/Bengaluru_House_Price_Prediction/blob/main/Bengaluru_House_Data.csv)

## Techstack
* Scikit-learn (for model building)
* Pandas (for dataframe)
* Numpy (handling data)
* Matplotlib (data visualization) 
## Description:
In this project we have used Linear Regression for making a linear model for floating value prediction
Approach used:
1.  Firstly install all the requirements for this project by using:
>!pip install requirements.txt
2.  Loading data: firstly we use pandas to read the csv file to a dataframe as:
>df1 = pd.read_csv("Bengaluru_House_Data.csv") <br/>
>df1.head()

![sample](https://github.com/lakshayd760/House_Price_Prediction/Images/Annotation%202023-09-19%20150948.png)

3. After preprocessing our data will look like:
![](https://github.com/lakshayd760/House_Price_Prediction/Images/Annotation%202023-09-19%20152105.png)
4.  After preprocessing the data we can train the model using:
> from sklearn.linearmodel import LinearRegression<br/>
> lr_clf = LinearRegression() <br/>
> lr_clf.fit(X_train,y_train)
5.  After we have build our model and trained on our training data, its time to test our data and make predictions and find our final score as :
> lr_clf.score(X_test,y_test)

    our model is giving us 86.29132245229445% accuracy we is good
6.  We can also perform cross validation to test model performance, in this case we are using K-fold cross validation with 5 folds as:
>from sklearn.model_selection import ShuffleSplit <br/>
>from sklearn.model_selection import cross_val_score <br/><br/>

>cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0) <br/>

>cross_val_score(LinearRegression(), X, y, cv=cv) <br/>
![](https://github.com/lakshayd760/Bengaluru_House_Price_Prediction/blob/main/Annotation%202023-09-19%20152351.png)

we can also make a model using deep learning architecture by using dense layers and for output we can use a single neuron with linear activation function 
