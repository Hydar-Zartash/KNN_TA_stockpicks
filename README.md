# KNN_TA_stockpicks

This program was an experiment to explore whether using various patterns could be used to predict stock behaviour where classical regression algorithms fail.
Given that technical indicators are often self fulfilling (stock with strong technical indicators will see an increase in demand from bots following the indicators), 
I decided to use the historical "context" of a stock to predict its future price. Using only indicators that are easily calculated using math based on recent stock price data.
Unfortunately, this meant neglecting the actual drivers of stock price change, news and irrational sentiments. Thus, the indicator is best used as a momentum predictor rather than actual performance. 
