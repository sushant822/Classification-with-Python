# Classification-with-Python

In this project, we explore various classification methods.

We train our model using our training dataset and then we test it using testing dataset.

Reports at the end show that our model is around 70% accurate.

First we load the data into pandas dataframe:

    df = pd.read_csv('data/loan_train.csv')

Upon veiwing the data, it looks like this:

![01_date_before](images/01_date_before.png)

As we can see, date and time needs to be an object for us to do analysis. We can use this code to fix it:

    df['due_date'] = pd.to_datetime(df['due_date'])
    df['effective_date'] = pd.to_datetime(df['effective_date'])

This is what it looks like now:

![02_date_after](images/02_date_after.png)