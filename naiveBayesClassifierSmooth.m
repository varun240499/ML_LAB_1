%% Machine Learning: Lab Assignment 1
% Soundarya Pallanti

% BUILDING A NAIVE BAYES CLASSIFIER WITH LAPLACE SMOOTHING
function [classification, error_rate] = naiveBayesClassifierSmooth(training_set, test_set, ground_truth)

    if nargin <2
        disp('Error: not enough input.\n');
        return 
    end
    
    [n, d] = size(training_set);
    [m, c] = size(test_set);
    
    % Checking number of coloumns
    if (d ~= c+1) 
        disp('Error: incorrect size of the sets.\n');
        return 
    end
    
    % Checking entries of the training set
    for i=1:n
        for j=1:d
            if (training_set(i,j) < 1)
                disp('Error: incorrect values of the trainig set.\n');
                return
            end
        end
    end
    
    % Checking entries of the test set
    for i=1:m
        for j=1:c
            if (test_set(i,j) < 1)
                disp('Error: incorrect values of the data set.\n');
                return
            end
        end
    end
    
    
    % TRAINING A NAIVE BAYES CLASSIFIER ON THE TRAINING SET
    % Computing marginal probabilty of X and a priori probability of H
    value_max = zeros(d,1);
    number_occurrences_x = zeros(n-1,d);
    marginal_prob_x = zeros(n-1,d);
    for i=1:d
       value_max(i) = training_set(1,i); % Maximum value of each feature, the first row of the training set
       for j=1:value_max(i)
          number_occurrences_x(j,i) = sum((training_set(2:n,i) == j));
          marginal_prob_x(j,i) = number_occurrences_x(j,i)/(n-1); % In the last column of this matrix there is the a priori probability of H
       end    
    end
    
    n_classes = value_max(d); % Number of classes

    % Computing the likelihood: P(X|H)=[(n° of x|h)+a]/[(n° of h)+an]
    likelihood = zeros(max(value_max),d-1,n_classes);
    a = 0.5; 
    for i=1:d-1 % For each feature
       for j=2:n % For each observation
           for k=1:value_max(i) % For each value
               for l=1:n_classes % For each class
                   if (training_set(j,i) == k && training_set(j,d) == l)
                       likelihood(k,i,l) = likelihood(k,i,l) + 1;
                   end
               end
           end
       end
    end
    for i=1:d-1
        for j=1:max(value_max)
            for k=1:n_classes
                likelihood(j,i,k) = (likelihood(j,i,k) + a)/(number_occurrences_x(k,d) + a*value_max(i));
            end
        end
    end
    %TRAINING ENDED
    
    % CLASSIFYING THE TEST SET ACCORDING TO THE INFERRED RULE
    % Computing a posteriori probability of H: P(H|X) = (likelihood*aPrioriProb)/partition_function
    posteriori_prob = ones(m-1,n_classes); 
    prob_feature = zeros(max(value_max),c,n_classes);
    partition = zeros(m-1,n_classes);
    classification = zeros(m-1,1);
    for k=1:n_classes % For each class
        for j=2:m % For each observation of the test set
            for i=1:c % For each feature
                value_testset = test_set(j,i);
                if (isnan(value_testset))
                else
                    prob_feature(j-1,i,k) = likelihood(value_testset,i,k);
                    posteriori_prob(j-1,k) = posteriori_prob(j-1,k) * prob_feature(j-1,i,k);
                end

            end
        end
        posteriori_prob(:,k) = posteriori_prob(:,k) * marginal_prob_x(k,d);
    end
    % Dividing for the partition function
    for i=1:m-1
        for k=1:n_classes
            partition(i) = partition(i) + posteriori_prob(i,k);
        end
        posteriori_prob(i,:) = posteriori_prob(i,:)/partition(i);
    end
    
    % Finding the maximum value of the a posteriori probability for each row of the test set
    for i=1:m-1
        for k=1:n_classes
            [t, classification(i,1)] = max(posteriori_prob(i,:));
        end
    end
    
    % Computing the error rate
    if nargin>2
        error_rate = (sum(classification ~= ground_truth))/(m-1);
    end

end