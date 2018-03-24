function [ w, gamm, obj, misclass ] = separateQP( dataMat, features, miu, quiet )
%SEPARATEQP Solves a quadratic program
% Requires CVX
% Preconditions:
%     `dataMat` the labels and features for all examples. It is a (m+k) x (1+n) matrix
%       where the first column is 1/0 for a positive/negative example respectively.
%     `features` the index of features to be separated on, each element in [1, n]
%     `miu` regularization term
%     `m` number of positive examples, 'malignant'
%     `k` number of negative examples, 
%     `n` length of one feature vector
%     `quiet` true if cvx shouldn't print anything, false otherwise
%     (default: true)
% Post-conditions:
%     `w` `gamm` separating plane is given by w * x + gamm
%     `gamma` intercept of separating plane
%     `obj` optimal objective value
%     `misclass` number of misclassifications in dataMat

X = dataMat(:, features + 1);
labels = dataMat(:, 1);

M = X(labels == 1, :);  % malignant features
B = X(labels == 0, :);  % benign features
n = length(features);  % size of a feature vector
m = size(M, 1);  % no. of malignant examples
k = size(B, 1);  % no. of benign examples

if ~exist('quiet', 'var'); quiet = true; end;
cvx_quiet(quiet);

cvx_begin
    variable w(n)
    variable gamm 
    variable y(m)
    variable z(k);
    minimize (1/m * ones(1, m) * y + 1/k * ones(1, k) * z + miu / 2 * w' * w)
    subject to
        M * w - gamm * ones(m, 1) + y >= ones(m, 1);
        -B * w + gamm * ones(k, 1) + z >= ones(k, 1);
        y >= 0;
        z >= 0;    
cvx_end

obj = cvx_optval;

predict = X*w - gamm > 0;
misclass = sum(predict~=labels);

end

