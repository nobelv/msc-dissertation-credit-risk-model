# A Cash Flow based model without jump risk to estimate Risk Premium

This is an implementation of the model described in the paper: "A CFO-based model of contingent claims with jump risk" by Nuno Silva, my MSc in Finance dissertation supervisor. 

It is a cash flow based credit risk model that is meant to estimate the risk premium implied in stock data. 
The model looks backwards in time and is not meant to be predictive in nature.

## Getting Started
### Prerequisites

This model was implemented using Python 3.6. 
It's using the following packages:
```
pandas
scipy.optimize
numpy
statsmodels.api
```

### Installing & Usage

To run this model you need only four files:

```
model_execute.py - to run the model
model_functions.py - the functions used in the model execution file
cashflow_based_statevar.csv - csv containing data as input for the model
na_treasury.csv - csv containing US treasury data as input for the model (10yr rate is used as proxy to risk free)
```

Then just run the model_execute file, making sure that model_functions.py is somewhere so it can be picked up in your PATH variable.

## Authors

**Victor Nobel** [nobelv](https://github.com/nobelv)

## Acknowledgments

* Nuno Silva for his invaluable support and wisdom.
* Everyone who has answered/endured my questions.

