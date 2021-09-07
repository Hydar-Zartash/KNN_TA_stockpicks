from sklearn import model_selection, neighbors
import yfinance as yf
import numpy as np
import pandas as pd

'''
Hydar Zartash
Dec 18, 2019

this program uses the sklearn KNN algorithm to identify patterns in technical indicators and
recent stock performance (obtained from yfinance) to "predict" whether a stock should go up by a given % amount in the
coming 30 days. This project was a simple inquiry into dataframes and KNN, technical indicators and past performance 
(and by extension this program) are useless for purposes of value investing. Furthermore, all indicators are those that can 
be calculated purely from stock data, which neglects sentiment and external factors.
'''

class Stock_KNN:
    """
    The object of this class includes a dataframe, a classifier trained on it 
    and some associated test and prediction stats 
    """    

    def __init__(self, ticker: str, target:int) -> None:
        """Initialize the object by downloading stock data and performing several methods

        Args:
            ticker (string): the ticker to be downloaded and used
        """        
        self.data = yf.download(ticker,start = "2002-11-02", end = "2021-08-28")
        self.target = target/100 + 1 #returns decimal value of int input (8% -> 1.08)

        self.RSI14()
        self.STOCHRSI()
        self.MACD()
        self.AROON()
        self.Williams()
        self.BULL()

        self.data = self.setup()

        self.accuracy = self.KNN()
        self.guesses = self.predict()

        self.indicator = self.growth_prediction()
        


        pass


    def RSI14(self) -> None:
        """The formula for relative strength index is as follows
        RSI = 100 - 100/(1- avg gain/avg loss)
        in general practice, > 70 indicates a selling oppurtunity and < 30 indicates a buy
        all averages used are simple moving averages
        """

        self.data['pct_change'] = 100*(self.data['Close']-self.data['Open'])/self.data['Open'] #daily perent change
    
        self.data['day_gain'] = self.data['pct_change'].where(self.data['pct_change'] > 0, other = 0) #filters the percent changes for only the gains
        self.data['avg_gain'] = self.data['day_gain'].rolling(window=14).mean() #take rolling avg of the gains

        self.data['day_loss'] = self.data['pct_change'].where(self.data['pct_change'] < 0,other = 0) #filters only precent changes of a loss
        self.data['avg_loss'] = self.data['day_loss'].rolling(window=14).mean() #takes rolling avg

        self.data['RSI-14'] = 100 -(100/ (1 - (self.data['avg_gain']/self.data['avg_loss']))) #unsmoothed RSI self.data

    def STOCHRSI(self) -> None:
        """stochastic RSI 
        calculate a stochastic oscillation for the RSI 14
        stoch = (current - recent min) / (recent max - recent min)
        """        
        self.data['STOCH-RSI'] = (self.data['RSI-14'] - self.data['RSI-14'].rolling(window=14).min())/(self.data['RSI-14'].rolling(window=14).max() - self.data['RSI-14'].rolling(window=14).min())

    def MACD(self) -> None:
        """moving average convergence divergence is another indicator calculated by short term EMA - long term EMA
        EMA is provided by the pandas.ewm().mean() method
        """        
        self.data['MACD'] = self.data['Close'].ewm(span = 12, adjust=False).mean() - self.data['Close'].ewm(span = 24, adjust=False).mean()

    def AROON(self) -> None:
        """the aroon oscillator is arron up - aroon down and measures the momentum
        Aroon up = 100*(interval length - days since rolling max on interval)/interval
        Aroon up = 100*(interval length - days since rolling min on interval)/interval
        we will be doing a 25 day interval
        """        
        self.data['AROON'] =  (100 * (25 - self.data['Close'].rolling(window=25).apply(np.argmax)) / 25)  - (100 * (25 - self.data['Close'].rolling(window=25).apply(np.argmin)) / 25)

    def Williams(self) -> None:
        """Williams R% is the (Highest high - Current close)/(Highest High - Lowest low)*-100%
        for any given range (14 in this case)
        """        
        self.data['R%'] = (self.data['High'].rolling(window=14).max() - self.data['Close']) / (self.data['High'].rolling(window=14).max() - self.data['Low'].rolling(window=14).min()) *-100

    def BULL(self) -> None:
        """Bull power is the formula of high - exponential weight average of the close
        """        
        self.data['Bull'] = self.data['High'] - self.data['Close'].ewm(span = 14, adjust=False).mean()
            
    def setup(self) -> pd.DataFrame:
        """ Adds a column to see if stock goes up by 8% in 30 days
         (1 for True, 0 for false)

        Returns:
            pd.DataFrame: returns df of cols: 
            Close Values
            RSI 14
            Stochastic oscillator of RSI 14
            MACD
            AROON
            Williams R%
            Bull-Bear indicator
            boolean whether stock grows X% in the next thirty days
        """        
        self.data['Shift'] = self.data['Adj Close'].shift(-30)
        self.data['Shift'] = self.data['Shift'].rolling(window=30).max()
        self.data['Growth X%'] = self.data['Adj Close']*self.target <= self.data['Shift']
        self.data['Growth X%'] = np.where(self.data['Growth X%']==True, 1, 0)

        final = self.data[['Adj Close', 'RSI-14', 'STOCH-RSI', 'MACD', 'AROON', 'R%', "Bull", 'Growth X%']]

        return final


    def KNN(self) -> float:
        """This method:
        1. removes the most recent 30 days from train and test data
        2. splits the growth column from the df and fillna with outlier
        3. train the classifier on 80% of the history
        4. test the accuracy of the classifier

        Returns:
            float: %accuracy of classifier
        """
        self.prediction_data = self.data[-30:]
        self.data = self.data[:-30]
        self.data.fillna(-99999, inplace=True)
        
        y = np.array(self.data['Growth X%'])
        X = np.array(self.data.drop(['Growth X%'], 1))

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
        self.clf = neighbors.KNeighborsClassifier()

        self.clf.fit(X_train, y_train)

        accuracy = self.clf.score(X_test, y_test)
        return accuracy*100 - 50

    def predict(self) -> pd.Series:
        """Return a series containing predictions for each day in the last 30
            1 => the stock will rise X% in 30 days
            0 => the stock will not rise by X%

        Returns:
            pandas.Series: returns a pandas series with guesses of if the stick will grow by X%
                    for each day's indiccators 
        """        
        return self.clf.predict(self.prediction_data.drop(['Growth X%'], 1))

    def growth_prediction(self) -> float:
        """This function returns an indicator from 0 to 100 indicating the reliability 
        with which the stock will rise 8% in 30 days

        In my opinion, an indicator value above 50% is a relatively safe bet to grow X%

        Returns:
            float: indicator 
        """        
        expected = 100 * (self.accuracy/100) * (sum(self.guesses)/len(self.guesses))
        return expected

stock = Stock_KNN('AMZN', 3)
print(f" \nIndicator Score: {stock.indicator} \nClassifier Accuracy: {stock.accuracy}")



