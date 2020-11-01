import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score


import warnings
warnings.filterwarnings("ignore")

from yahoo_fin import stock_info as si

validation_scores_allticks = []

def get_last_stock_price(ticker, last=False):
    #if last:
    #    now = datetime.now()
    #    start_date = now - timedelta(days=30)
    #    return si.get_data(ticker, start_date)
    return si.get_data(ticker)


def features_creation(df):
    daily_change_OC = []

    for i in range(len(df)):
        daily_change_OC.append(100 * (df.adjclose.iloc[i] - df.open.iloc[i]) / (df.open.iloc[i]))

    df['daily percentage change'] = daily_change_OC
    daily_pct_change = df['daily percentage change'].pct_change()
    df['vol_dailychange'] = daily_pct_change

    volume_change = df['volume'].pct_change()
    df['volume_change'] = volume_change

    df = df.fillna(0)

    df['ma8'] = df['adjclose'].rolling(window=8).mean()
    df['ma12'] = df['adjclose'].rolling(window=12).mean()
    df['ma20'] = df['adjclose'].rolling(window=20).mean()
    df['ma50'] = df['adjclose'].rolling(window=50).mean()

    df['std20'] = df['adjclose'].rolling(window=20).std()
    df['skew20'] = df['adjclose'].rolling(window=20).skew()
    df['kurt20'] = df['adjclose'].rolling(window=20).kurt()

    df['auto1'] = df['adjclose'].rolling(1).apply(lambda x: pd.Series(x).autocorr())
    df['auto2'] = df['adjclose'].rolling(2).apply(lambda x: pd.Series(x).autocorr())
    df['auto3'] = df['adjclose'].rolling(3).apply(lambda x: pd.Series(x).autocorr())
    df['auto4'] = df['adjclose'].rolling(4).apply(lambda x: pd.Series(x).autocorr())
    df['auto5'] = df['adjclose'].rolling(5).apply(lambda x: pd.Series(x).autocorr())

    vol30 = df['vol_dailychange'].rolling(30).std() * np.sqrt(30)
    vol90 = df['vol_dailychange'].rolling(90).std() * np.sqrt(90)
    df['30D_vol'] = vol30
    df['90D_vol'] = vol90

    df = df.fillna(0)

    return df


def label_creation(df):
    buy_or_sell_labels = []

    for i in range(len(df)):
        if df['daily percentage change'].iloc[i] > 0:
            buy_or_sell_labels.append(1)
        if df['daily percentage change'].iloc[i] <= 0:
            buy_or_sell_labels.append(-1)

    df['labels'] = buy_or_sell_labels
    return df


def preprocessing(df):
    scaler = MinMaxScaler()
    columns = ['adjclose', 'ma8', 'ma12', 'ma20',
               'ma50']

    for col in columns:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    return df

class TimeBasedCV(object):
    '''
    Parameters
    ----------
    train_period: int
        number of time units to include in each train set
        default is 30
    test_period: int
        number of time units to include in each test set
        default is 7
    freq: string
        frequency of input parameters. possible values are: days, months, years, weeks, hours, minutes, seconds
        possible values designed to be used by dateutil.relativedelta class
        default is days
    '''

    def __init__(self, train_period=100, test_period=50, freq='days'):
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq

    def split(self, data, validation_split_date=None, date_column='timestamp', gap=0):
        '''
        Generate indices to split data into training and test set

        Parameters
        ----------
        data: pandas DataFrame
            your data, contain one column for the record date
        validation_split_date: datetime.date()
            first date to perform the splitting on.
            if not provided will set to be the minimum date in the data after the first training set
        date_column: string, deafult='record_date'
            date of each record
        gap: int, default=0
            for cases the test set does not come right after the train set,
            *gap* days are left between train and test sets

        Returns
        -------
        train_index ,test_index:
            list of tuples (train index, test index) similar to sklearn model selection
        '''

        # check that date_column exist in the data:
        try:
            data[date_column]
        except:
            raise KeyError(date_column)

        train_indices_list = []
        test_indices_list = []

        if validation_split_date == None:
            validation_split_date = data[date_column].min().date() + eval(
                'relativedelta(' + self.freq + '=self.train_period)')

        start_train = validation_split_date - eval('relativedelta(' + self.freq + '=self.train_period)')
        end_train = start_train + eval('relativedelta(' + self.freq + '=self.train_period)')
        start_test = end_train + eval('relativedelta(' + self.freq + '=gap)')
        end_test = start_test + eval('relativedelta(' + self.freq + '=self.test_period)')

        while end_test < data[date_column].max().date():
            # train indices:
            cur_train_indices = list(data[(data[date_column].dt.date >= start_train) &
                                          (data[date_column].dt.date < end_train)].index)

            # test indices:
            cur_test_indices = list(data[(data[date_column].dt.date >= start_test) &
                                         (data[date_column].dt.date < end_test)].index)

            print("Train period:", start_train, "-", end_train, ", Test period", start_test, "-", end_test,
                  "# train records", len(cur_train_indices), ", # test records", len(cur_test_indices))

            train_indices_list.append(cur_train_indices)
            test_indices_list.append(cur_test_indices)

            # update dates:
            start_train = start_train + eval('relativedelta(' + self.freq + '=self.test_period)')
            end_train = start_train + eval('relativedelta(' + self.freq + '=self.train_period)')
            start_test = end_train + eval('relativedelta(' + self.freq + '=gap)')
            end_test = start_test + eval('relativedelta(' + self.freq + '=self.test_period)')

        # mimic sklearn output
        index_output = [(train, test) for train, test in zip(train_indices_list, test_indices_list)]

        self.n_splits = len(index_output)

        return index_output

    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

def main_function(ticker):

    df = get_last_stock_price(ticker)
    df1 = get_last_stock_price(ticker)
    df = df.reset_index()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'adjclose', 'volume', 'ticker']

    df['timestamp'] = pd.to_datetime(df.timestamp)

    start_date = '2018-01-01'
    end_date = '2020-01-01'

    df = features_creation(df)
    df = label_creation(df)
    preprocessing(df)
    df = df[df.timestamp > start_date]  # Start Date
    df_train_test = df[df.timestamp < end_date]  # End Date

    XY_train = df_train_test.loc[:,
               ['timestamp', 'adjclose', 'ma8', 'ma12', 'ma20', 'std20', 'skew20', 'kurt20',
    '30D_vol', '90D_vol', 'auto1', 'auto2', 'auto3', 'auto4', 'auto5', 'ma50','labels']]

    data_for_modeling = XY_train
    tscv = TimeBasedCV(train_period=20,
                       test_period=5,
                       freq='days')
    for train_index, test_index in tscv.split(data_for_modeling, validation_split_date=datetime.date(2018, 1, 21),
                                              date_column='timestamp'):
        # get number of splits
        print(train_index, test_index)

    tscv.get_n_splits()

    #### Example- compute average test sets score: ####
    X = data_for_modeling[['timestamp', 'adjclose', 'ma8', 'ma12', 'ma20', 'std20', 'skew20', 'kurt20',
                           '30D_vol', '90D_vol','auto1','auto2','auto3','auto4','auto5','ma50']]
    y = data_for_modeling['labels']

    ##############################

    scores = []

    for train_index, test_index in tscv.split(X, validation_split_date=datetime.date(2018, 1, 21)):
        data_train = X.loc[train_index].drop('timestamp', axis=1)
        target_train = y.loc[train_index]

        data_test = X.loc[test_index].drop('timestamp', axis=1)
        target_test = y.loc[test_index]

        # if needed, do preprocessing here

        # Create a Gaussian Classifier
        clf = RandomForestClassifier(n_estimators=50,criterion = 'gini', max_depth = 5,
                                     min_samples_split = 2, min_samples_leaf = 1,max_features = 'sqrt')

        # param_grid = {
        #      'n_estimators': [50,100,150],
        #      'max_features': ['auto', 'sqrt'],
        #      'criterion' :  ['gini', 'entropy'],
        #      'max_depth' : [None,5,10,15]}


        # Train the model using the training sets y_pred=clf.predict(X_test)
        #clf.fit(data_train, target_train)
        # clf_opt = GridSearchCV(clf, param_grid=param_grid, scoring=f1_scorer, cv=5)
        clf.fit(data_train, target_train)

        preds = clf.predict(data_test)
        scores.append(balanced_accuracy_score(target_test, preds))

    average_BAscore = np.mean(scores)
    print("Balanced Accuracy: %f" % (100 * average_BAscore))
    # print("Best Parameters: %s" % (clf_opt.best_params_))

    predictions = []
    true_inputs = []
    validation_scores = []

    df_valid = df[df.timestamp > end_date]
    df_valid = df_valid.drop(['timestamp'], axis=1)
    for i in range(len(df_valid)):
        val = df_valid.loc[:, ['adjclose', 'ma8', 'ma12', 'ma20', 'std20', 'skew20', 'kurt20',
                               '30D_vol', '90D_vol', 'auto1', 'auto2', 'auto3', 'auto4', 'auto5', 'ma50']]
        pred_input = clf.predict(val.iloc[i].values.reshape(1, -1))
        predictions.append(int(pred_input[0]))

        true = df_valid.loc[:, ['labels']]
        true_input = true.iloc[i].values

        true_inputs.append(int(true_input[0]))
        validation_scores.append(balanced_accuracy_score(true_input, pred_input))

    print("Validation Balanced Accuracy: %f" % (100 * np.mean(validation_scores)))
    validation_scores_allticks.append(np.mean(validation_scores))

    if pred_input == 1:
        print('Tomorrow Predicition: BUY')
        positions = "BUY"
    elif pred_input == -1:
        print('Tomorrow Predicition: SELL')
        positions = "SELL"

    last_price = df1.adjclose.iloc[-1]
    latest_price = "{:.2f}".format(last_price)
    average_validation_score = (100 * np.mean(validation_scores))
    avg_vscore = "{:.2f}".format(average_validation_score)

    return positions, latest_price, avg_vscore



