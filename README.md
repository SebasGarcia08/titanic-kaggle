# titanic-kaggle
Kaggle first submission

## Context
Analysis of data

Titanic ML competition

Use machine learning to create a model that predicts which
passengers survived the Titanic shipwreck

On April 15, 1912, during her maiden voyage, the Titanic,
widely considered "unsinkable", she sank after
collide with an iceberg. Unfortunately, there was no
enough lifeboats for everyone on board, which resulted
in the death of 1502 of the 2224 passengers and crew.
Create a predictive model that answers the question: "What
type of people were most likely to survive? "
using passenger data (i.e. name, age, gender,
socioeconomic class, etc.).

## ANALYSIS QUESTIONS
We are going to use only the train.csv file
### Complexity 1 (35%)
1. Load the train.csv dataset with index_col for PassengerId
2. Convert column names to lowercase
3. Explore the dataset and identify the variables that have missing data.
4. Transform the variables to the following types (only if necessary):
1. Name -> object
2. Ticket -> object
3. Cabin -> category
4. Embarked -> category
5. pclass -> category
5. Filter the data (evaluate if by rows or columns) and obtain the records that are not
they have missing data
6. What is the largest number of records (values) for the Survived variable?
(0.1)
7. Who survived more, women or men?
8. Calculate the average passenger fare for class 1 (pclass)

### Complexity 2 (35%)

9. Is there something particular between
the age of the people and the
variable Survived?

10. Do large families
survive longer? SibSp +
Parch

11. For this data set,
How true is it that “Las
women and children
first…"?

Complexity 3 (30%)

12. Develop a model that
allow to obtain the values
of the target variable

13. I sent the results of the
file test.csv to
Kaggle platform, take
capture of evidence and
attach as picture to
jupyter notebook