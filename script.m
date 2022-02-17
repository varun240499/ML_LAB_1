%% Machine Learning: Lab Assignment 1
% Soundarya Pallanti

clear

%% DATA PREPROCESSING
% Uploading dataset
load('Weather_dataset.mat');
dataset = table2array(Weatherdataset);
[n, d] = size(dataset);

% Splitting into training and test sets
index = randperm(n);
m = 10; % Dimension of the training set
training_set = dataset(index(1:m), :);
test_set = dataset(index(m+1:end), 1:(d-1));
ground_truth = dataset(index(m+1:end), d);
         
%% NAIVE BAYES CLASSIFIER
% Calling the Naive Bayes classifier
[classification, error_rate] = naiveBayesClassifier(training_set, test_set, ground_truth);

% Printing the results
for i=1:(n-m)
   fprintf('The prediction is %d and the corresponding target is %d\n', classification(i), ground_truth(i));
end
fprintf('The error rate is: %f\n', error_rate);

%% LAPLACE SMOOTHING
% Adding to both training and test set the information about the number of levels
value_max = zeros(d,1); % Number of maximum value for each feature
for i=1:d
    value_max(i) = max(dataset(:,i));
end
training_set_pro = [value_max'; training_set];
test_set_pro = [value_max(1:d-1)'; test_set];

% Calling the Naive Bayes classifier with Laplace smoothing
[classification_smooth, error_rate_smooth] = naiveBayesClassifierSmooth(training_set_pro, test_set_pro, ground_truth);

% Printing the results obtained with the Laplace smoothing
for i=1:(n-m)
   fprintf('The prediction with the Laplace smoothing is is %d and the corresponding target is %d\n', classification_smooth(i), ground_truth(i));
end
fprintf('The error rate with the Laplace smoothing is: %f\n', error_rate_smooth);



%% MUSHROOM DATASET
%Uploading dataset
load('Mushroom_dataset.mat');
dataset2 = table2array(mushroom);

% Moving the class column in the last column of the matrix and deleting the 11st feature (12 column) because inconsistent
dataset2 = [dataset2(:,2:11), dataset2(:,13:end), dataset2(:,1)];
[l, c] = size(dataset2);

% Splitting into training and test sets
index = randperm(l);
m = 5000; % Dimension of the training set
training_set2 = dataset2(index(1:m), :);
test_set2 = dataset2(index(m+1:end), 1:(c-1));
ground_truth2 = dataset2(index(m+1:end), c);

%% NAIVE BAYES CLASSIFIER
% Calling the Naive Bayes classifier
[classification2, error_rate2] = naiveBayesClassifier(training_set2, test_set2, ground_truth2);

% Printing the results
fprintf('The error rate of mushroom is: %f\n', error_rate2);

%% LAPLACE SMOOTHING
% Adding to both training and test set the information about the number of levels
value_max2 = [6, 4, 10, 2, 9, 4, 3, 2, 12, 2, 4, 4, 9, 9, 2, 4, 3, 8, 9, 6, 7, 2]'; % Taken from the data description
training_set_pro2 = [value_max2'; training_set2];
test_set_pro2 = [value_max2(1:c-1)'; test_set2];

% Calling the Naive Bayes classifier with Laplace smoothing
[classification_smooth2, error_rate_smooth2] = naiveBayesClassifierSmooth(training_set_pro2, test_set_pro2, ground_truth2);

% Printing the results obtained with the Laplace smoothing
fprintf('The error rate of mushroom with the Laplace smoothing is: %f\n', error_rate_smooth2);
