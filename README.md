# A cash flow based model to compute a risk pricing indicator (without jump risk).

This code represents an implementation of the model described in the paper: "A CFO-based model of contingent claims with jump risk" by Nuno Silva, my MSc in Finance dissertation supervisor. There are some modifications compared to the main model, the main difference being that we ommit jump risk.

In its essence it is a cash flow based credit risk model that is meant to compute an indicator of how risk is priced in the market as implied by stock data. The model looks backwards in time and is not meant to be predictive in nature. I might share the final version of my master thesis here, which provides further clarifcation, upon formal completion of the program.

## Getting Started
### Prerequisites

This model was implemented using Python 3.6. 
It's using the following packages:
```
statsmodels.api
scipy.optimize
scipy.stats
matplotlib
seaborn
pandas
numpy
```

### Installing & Usage

To run this model you need only three files:

```
model_execute.py - to run the model
model_functions.py - the functions used in the model execution file
cashflow_data.csv - csv containing data as input for the model
```

Then just execute the model_execute script, making sure that model_functions.py is somewhere so it can be picked up in your PATH variable.

## Authors

**Victor Nobel** [nobelv](https://github.com/nobelv)

## Acknowledgments

* Nuno Silva for his invaluable support and wisdom.
* Everyone who has answered/endured my questions.

