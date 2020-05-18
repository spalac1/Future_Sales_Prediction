# Future_Sales_Prediction
Initial practice with the kaggle dataset to use timeseries to predict sales of items by shop on a certain day

Steps I took during this work:
Did some standard plotting with Seaborn in order to get an idea of the shape of the data I was working with.
This led to me finding that there were some pretty clear outliers (I arbitrarily chose 10 SDs), which I then decided to remove during inital 
	data cleaning and replot in order to get a visual of the newly cleaned data.
Then, as usual, I took a basic linear model to get an idea on the predictive power of the simplest model for a baseline