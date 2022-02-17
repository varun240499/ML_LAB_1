# Lab assignment 1: Naive Bayes classifier
- Task 1: Data preprocessing
- Task 2: Build a naive Bayes classifier
- Task 3: Improve the classifier with Laplace (additive) smoothing
Describe everything in a report.

## Task 1: Data preprocessing
Here you have to download data and prepare them for being used with a Matlab program.

Download the Weather data set and the Weather data description.

You can also use a larger (8124 x 22) dataset to classify mushrooms as edible/poisonous (description).

Remark: For use with Matlab it is convenient to convert all attribute values (called "levels" when they are nominal) into integers>=1. So they can be used to index matrices. You can do so by either using a plain text editor, or reading the data into a spreadsheet table and then saving it back (remember to "save as" a plain text file or at most a .csv file, not a spreadsheet file).

## Task 2: Build a naive Bayes classifier
Here you have to create a program (a Matlab function for instance) that takes the following parameters:

1) a set of data, as a n x (d+1) matrix, to be used as the training set
2) another set of data, as a m x d matrix, to be used as the test set
3) OPTIONALLY, another set of data, as a m x 1 matrix, to be used as the test set ground truth (class labels)

The program should:

1) Check that the number of columns of the second matrix equals the number d of columns of the first matrix – 1
2) Check that no entry in any of the two data sets is <1
3) Train a Naive Bayes classifier on the training set (first input argument), using its last column as the target
4) Classify the test set according to the inferred rule, and return the classification obtained
5) If the test set has the optional additional column, use this as a target, compute and return the error rate obtained (number of errors / m)

Write a script that, without the user's intervention:

1) loads the weather data set (already converted in numeric form)
2) splits it into a training set with 10 randomly chosen patterns and a test set with the remaining 4 patterns
3) calls the naive Bayes classifier and reads the results
4) prints the results: classification for each pattern of the test set, corresponding target if present, error rate if computed.

Remarks: The classifier should be programmed in such a way to be suitable for the following situations:

- other data sets of different cardinalities (n, m) and dimensionality (d)
- test sets not including a target
- test sets including attribute values that were not in the training set.
In the last case the program should issue an error and discard the corresponding pattern, because if you use that value to index a matrix it will likely go out of bounds (e.g., if you computed probabilities for values 1, 2 and 3 and you get value 4 in the test).

Use the slides to implement the classifier. Note that all attributes are categorical, but some are non-binary having more than 2 levels. Your code should not assume that any of the attributes or the class are binary.

## Task 3: Improve the classifier with Laplace (additive) smoothing
It may be the case that some values of some attribute do not appear in the training set, however you know the number of levels in advance.

An example is binary attributes (e.g., true/false, present/absent...). You know that attribute x can be either true or false, but in your training set you only have observations with x = false. This does not mean that the probability of x = true is zero!!!

To deal with this case, you should change your code in 2 ways:

1) In the data preparation step, add the information about the number of levels. This means that for each data column you should add the number of possible different values for that column.

In the case of the weather data, this can be a list (or vector) [ 3, 3, 2, 2 ]. You can add this list as a first row in all the data sets (training and test), to be interpreted with this special meaning.

2) When you compute probabilities, you should introduce additive smoothing or Laplace smoothing in your formulas. This means adding some terms that account for your prior belief. Since you don't know anything, your prior belief is that all values are equally probable. In this case, Laplace smoothing gives probability = 1/2 (for binary variables) or more generally probability = 1/n (for variables with n possible values).

In formulas, you should replace:

P(attribute x = value y | class z) =
   (number of observations in class z where attribute x has value y) / (number of observations in class z)
with the following:

P(attribute x = value y | class z) =
   ((number of observations in class z where attribute x has value y) +a) / ((number of observations in class z) +an)
where:

n is the number of values of attribute x
you can use a = 1. As a further refinement, you can experiment with a > 1 (which means "I trust my prior belief more than the data") or with a < 1 (which means "I trust my prior belief less than the data")