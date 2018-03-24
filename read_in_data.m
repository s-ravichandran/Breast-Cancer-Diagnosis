clear;

input_file = 'input_data/wdbc.data';
dataDim = 30;  % size of a feature vector
rng('default');
quiet = true;
divider = repmat('*', [1, 30]);

%% 1a

miu = 0.001;  % regularization term
fracTest = 0.1;
reord = 0;  % random reordering of data
[train,~,~,~] = wdbcData(input_file,dataDim,fracTest,reord);
[w, gamm, obj, misclass] = separateQP(train, 1:dataDim, miu, quiet);

disp(divider);
fprintf('miu = %f, fracTest=%f, reord=%i\n', miu, fracTest, reord);
disp('w = ');
disp(w');
fprintf('gamma = %f\n', gamm);
fprintf('optimal objective value = %f\n', obj);
fprintf('Number of training misclassifications = %i\n', misclass);

%% 1b

miu = 0.001;  % regularization term
fracTest = 0.15;
reord = 1;  % random reordering of data
[train,~,~,~] = wdbcData(input_file,dataDim,fracTest,reord);
[w, gamm, obj, misclass] = separateQP(train, 1:dataDim, miu, quiet);

disp(divider);
fprintf('miu = %f, fracTest=%f, reord=%i\n', miu, fracTest, reord);
disp('w = ');
disp(w');
fprintf('gamma = %f\n', gamm);
fprintf('optimal objective value = %f\n', obj);
fprintf('Number of training misclassifications = %i\n', misclass);

%% 2

miu = 0.001;  % regularization term

fracTests = [0.1 0.15 0.2 0.05];
reords = [0 0 1 1];
for testNum = 1 : length(fracTests)
    fracTest = fracTests(testNum);
    reord = reords(testNum);
    
    [train,test,ntrain,ntest] = wdbcData(input_file,dataDim,fracTest,reord);
    [w, gamm, ~, misclassTrain] = separateQP(train, 1:dataDim, miu, quiet);

    XTest = test(:, 2:end);
    yTest = test(:, 1);
    predict = XTest*w - gamm > 0;
    misclassTest = sum(predict~=yTest);
    
    disp(divider);
    fprintf('fracTest=%f, reord=%i\n', fracTest, reord);
    fprintf('Number of test misclassifications = %i\n', misclassTest);
end

%% 3

miu = 0.0008;  % regularization term

fracTest = 0.12;
reord = 0;
[train,test,ntrain,ntest] = wdbcData(input_file,dataDim,fracTest,reord);
bestMisclassTrain = ntrain + 1;  % nothing can be worse than this
bestFeatures = [];
for features = nchoosek(1:dataDim, 2)'
    
    [w, gamm, ~, misclassTrain] = separateQP(train, features, miu, quiet);
    if misclassTrain < bestMisclassTrain
        bestMisclassTrain = misclassTrain;
        bestFeatures = features;
        fprintf('attributes %2d %2d: misclass %3d\n', ...
            features(1), features(2), misclassTrain);
    end
    
end

%% 4

[w, gamm, ~, misclassTrain] = separateQP(train, bestFeatures, miu, quiet);
XTest = test(:, bestFeatures + 1);
yTest = test(:, 1);
predict = XTest*w - gamm > 0;
misclassTest = sum(predict~=yTest);

disp(divider);
fprintf('Test misclassification = %i\n', misclassTest);

figure; hold on;
plot(XTest(yTest==1, 1), XTest(yTest==1, 2), '+');
plot(XTest(yTest==0, 1), XTest(yTest==0, 2), 'o');
slope = -w(1)/w(2);
intercept = gamm/ w(2);
refline(slope, intercept);
xlabel(sprintf('Feature %i', bestFeatures(1)));
ylabel(sprintf('Feature %i', bestFeatures(2)));
legend('Malignant test examples', 'Benign test examples');
title('Test set classification results');
