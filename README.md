# Predicting Urban Walkability: Identifying Key Factors and Opportunities for Improvement
Ryan Soucy, Kaia Quigg


## Research Goals:
Our main research goal is to attempt to predict the walkability of a city based on several
predictors and to determine which of the predictors appear to be most impactful in our model.
We would then try to determine which cities may be best candidates to increase their walkability
scores based on our most impactful predictors.

## Methodology

* Data cleaning:

Initial data cleaning will involve removing ~142,000 null values found in ~53,000 rows,
deleting repeat and highly correlated predictors, and conducting principal component analysis as
a way to feature select and reduce dimensionality. This may be a fairly manual effort based on
our dataset and the number of predictors present. We then plan to query a random subset for each
model iteration in order to make use of all data points while maintaining favorable run times and
processing speeds.

* Model Selection:
  
We will then use three different machine learning models to determine which model may
best predict the walkability scores. The three models we plan to use are random forest, neural
networks, and support vector machines. These three models are known for their ability to handle
datasets with large dimensionality. Before each model, we will preprocess the data accordingly
and be sure to avoid data leakage. Each model will be tuned using various tuning parameters and
GridSearchCV .

* Linear regression and classification:
  
We aim to explore both linear regression and classification as methods of identifying the
most impactful walkability attributes. Classification may be a viable option as our target variable,
Walkability Index, is a score of 1-20 but is categorized in four groups. (Most walkable, Above
average walkable, Below average walkable, Least walkable)

* Decision Trees and Random Forest:

We will use decision trees to build our machine learning model in order to examine which
cities have the lowest walkability. We will use libraries such as scikit-learn or XGBoost for
building and fine-tuning. For cities scoring in the lowest walkability categories, we will analyze
the attribute thresholds identified by decision tree splits to suggest actionable changes that would
shift these cities closer to the next walkability index group.Visualization may be particularly
useful with this step.

## Technical Information:
* Programming Languages: Python

* Libraries Used:

    * Data Processing: pandas, NumPy

    * Feature Engineering: PCA (scikit-learn)

    * Machine Learning Models: scikit-learn (Random Forest, Decision Trees, SVM), TensorFlow/Keras (Neural  Networks), XGBoost

    * Hyperparameter Tuning: GridSearchCV

    * Visualization: Matplotlib, Seaborn

## Possible outcomes and Implications:

Possible outcomes for our project are highly variable and highly dependent on our initial
findings when building our machine learning models. The initial goal will be to accurately
predict walkability scores.

The potential implications of this project are numerous. Aside from providing a deeper
understanding of what truly contributes to a walkable environment, there is the potential to
uncover trends that point to either simple fixes that a city can implement, or deeper systemic
issues that have been overlooked.

## Conclusion:

This project combines many machine learning techniques in the hopes of uncovering
walkability trends across diverse urban environments. As most features in this dataset surround
demographical data, the resulting trends may show that solutions extend beyond simple
infrastructure fixes that policy makers could implement and instead delve into an examination of
broader social and economic implications of walkability.
# Capstone557
