% Clear workspace, close all figures, and clear command window
clear; close all; clc;

% Load data from Excel file
M = readmatrix('parametric_study.xlsx');

% Define input (x) and target (t) data
x = M(:, 1:end-1)';
t = M(:, end)';

% Normalize data to [-1,1]
[x_norm, ps_input] = mapminmax(x, -1, 1);
[t_norm, ps_output] = mapminmax(t, -1, 1);

% Determine size of dataset and split indices
Q = size(x_norm, 2);
Q1 = floor(Q * 0.70); % 70% for training
Q2 = floor(Q * 0.15); % 15% for validation
Q3 = Q - Q1 - Q2;    % 15% for testing
ind = randperm(Q);

% Create indices for each set
indTrain = ind(1:Q1);
indVal = ind(Q1+1:Q1+Q2);
indTest = ind(Q1+Q2+1:end);

% Split normalized data
xTrain = x_norm(:, indTrain);
tTrain = t_norm(:, indTrain);
xVal = x_norm(:, indVal);
tVal = t_norm(:, indVal);
xTest = x_norm(:, indTest);
tTest = t_norm(:, indTest);

% Initialize arrays to store performance metrics
maxNeurons = 40;
minNeurons = 1;
neurons = minNeurons:maxNeurons;

% Initialize arrays for both algorithms
mse_train_lm = zeros(1, maxNeurons);
mse_val_lm = zeros(1, maxNeurons);
mse_test_lm = zeros(1, maxNeurons);
mse_train_br = zeros(1, maxNeurons);
mse_test_br = zeros(1, maxNeurons);
mse_val_br = zeros(1, maxNeurons);

% Store best networks and their properties
best_net_lm = [];
best_net_br = [];
best_mse_lm = Inf;
best_mse_br = Inf;
best_neurons_lm = 0;
best_neurons_br = 0;

% Store predictions for best networks
best_train_pred_lm = [];
best_val_pred_lm = [];
best_test_pred_lm = [];
best_train_pred_br = [];
best_test_pred_br = [];
best_val_pred_br = [];

% Arrays to store RMSE and R^2
rmse_train_lm = zeros(1, maxNeurons);
rmse_val_lm   = zeros(1, maxNeurons);
rmse_test_lm  = zeros(1, maxNeurons);

rmse_train_br = zeros(1, maxNeurons);
rmse_test_br  = zeros(1, maxNeurons);
rmse_val_br   = zeros(1, maxNeurons);

r2_train_lm = zeros(1, maxNeurons);
r2_val_lm   = zeros(1, maxNeurons);
r2_test_lm  = zeros(1, maxNeurons);

r2_train_br = zeros(1, maxNeurons);
r2_test_br  = zeros(1, maxNeurons);
r2_val_br   = zeros(1, maxNeurons);

% Loop through different numbers of neurons
for i = neurons
    fprintf('Training networks with %d neurons\n', i);
    
    % Train Levenberg-Marquardt network with validation
    net_lm = feedforwardnet(i, 'trainlm');
    net_lm.layers{1}.transferFcn = 'tansig';
    net_lm.layers{2}.transferFcn = 'purelin';
    net_lm.divideFcn = 'divideind';
    net_lm.divideParam.trainInd = 1:size(xTrain,2);
    net_lm.divideParam.valInd = 1:size(xVal,2);
    net_lm.divideParam.testInd = [];
    net_lm = configure(net_lm, [xTrain xVal], [tTrain tVal]);
    [net_lm, tr_lm] = train(net_lm, [xTrain xVal], [tTrain tVal]);
    
    % Calculate MSE for LM (in normalized space)
    yTrain_lm_norm = net_lm(xTrain);
    yVal_lm_norm = net_lm(xVal);
    yTest_lm_norm = net_lm(xTest);
    
    % Transform predictions back to original scale
    yTrain_lm = mapminmax('reverse', yTrain_lm_norm, ps_output);
    yVal_lm = mapminmax('reverse', yVal_lm_norm, ps_output);
    yTest_lm = mapminmax('reverse', yTest_lm_norm, ps_output);
    
    % Calculate MSE in original scale
    mse_train_lm(i) = mean((yTrain_lm - t(:,indTrain)).^2);
    mse_val_lm(i) = mean((yVal_lm - t(:,indVal)).^2);
    mse_test_lm(i) = mean((yTest_lm - t(:,indTest)).^2);
    
    % Update best LM network if necessary (based on validation performance)
    if mse_val_lm(i) < best_mse_lm
        best_mse_lm = mse_val_lm(i);
        best_net_lm = net_lm;
        best_neurons_lm = i;
        best_train_pred_lm = yTrain_lm;
        best_val_pred_lm = yVal_lm;
        best_test_pred_lm = yTest_lm;
    end
    
    % Train Bayesian Regularization network
    net_br = feedforwardnet(i, 'trainbr');
    net_br.layers{1}.transferFcn = 'tansig';
    net_br.layers{2}.transferFcn = 'purelin';
    net_br.divideFcn = 'divideind';
    net_br.divideParam.trainInd = 1:size(xTrain,2);
    net_br.divideParam.valInd = 1:size(xVal,2);
    net_br.divideParam.testInd = [];
    net_br = configure(net_br, [xTrain xVal], [tTrain tVal]);
    [net_br, tr_br] = train(net_br, [xTrain xVal], [tTrain tVal]);
    
    % Calculate MSE for BR (in normalized space)
    yTrain_br_norm = net_br(xTrain);
    yTest_br_norm = net_br(xTest);
    yVal_br_norm = net_br(xVal);

    % Transform predictions back to original scale
    yTrain_br = mapminmax('reverse', yTrain_br_norm, ps_output);
    yTest_br = mapminmax('reverse', yTest_br_norm, ps_output);
    yVal_br = mapminmax('reverse', yVal_br_norm, ps_output);

    % Calculate MSE in original scale - FIXED: Was overwriting LM training MSE
    mse_train_br(i) = mean((yTrain_br - t(:,indTrain)).^2);
    mse_test_br(i) = mean((yTest_br - t(:,indTest)).^2);
    mse_val_br(i) = mean((yVal_br - t(:,indVal)).^2);

    % Update best BR network
    if mse_val_br(i) < best_mse_br
        best_mse_br = mse_val_br(i);
        best_net_br = net_br;
        best_neurons_br = i;
        best_train_pred_br = yTrain_br;
        best_test_pred_br = yTest_br;
        best_val_pred_br = yVal_br;
    end

    nTrainCols = size(xTrain,2);
    nValCols   = size(xVal,2);

    % Compute RMSE from MSE
    rmse_train_lm(i) = sqrt(mse_train_lm(i));
    rmse_val_lm(i)   = sqrt(mse_val_lm(i));
    rmse_test_lm(i)  = sqrt(mse_test_lm(i));

    rmse_train_br(i) = sqrt(mse_train_br(i));
    rmse_test_br(i)  = sqrt(mse_test_br(i));
    rmse_val_br(i)   = sqrt(mse_val_br(i));

    % Compute R² for LM
    y_true_train_lm = t(:, indTrain);
    ss_res_lm_train = sum((yTrain_lm - y_true_train_lm).^2);
    ss_tot_lm_train = sum((y_true_train_lm - mean(y_true_train_lm)).^2);
    r2_train_lm(i)  = 1 - (ss_res_lm_train / ss_tot_lm_train);
    
    y_true_val_lm = t(:, indVal);
    ss_res_lm_val = sum((yVal_lm - y_true_val_lm).^2);
    ss_tot_lm_val = sum((y_true_val_lm - mean(y_true_val_lm)).^2);
    r2_val_lm(i)   = 1 - (ss_res_lm_val / ss_tot_lm_val);
    
    y_true_test_lm = t(:, indTest);
    ss_res_lm_test = sum((yTest_lm - y_true_test_lm).^2);
    ss_tot_lm_test = sum((y_true_test_lm - mean(y_true_test_lm)).^2);
    r2_test_lm(i)  = 1 - (ss_res_lm_test / ss_tot_lm_test);
    
    % Compute R² for BR 
    y_true_train_br = t(:, indTrain);
    ss_res_br_train = sum((yTrain_br - y_true_train_br).^2);
    ss_tot_br_train = sum((y_true_train_br - mean(y_true_train_br)).^2);
    r2_train_br(i)  = 1 - (ss_res_br_train / ss_tot_br_train);
    
    y_true_val_br = t(:, indVal);
    ss_res_br_val = sum((yVal_br - y_true_val_br).^2);
    ss_tot_br_val = sum((y_true_val_br - mean(y_true_val_br)).^2);
    r2_val_br(i)   = 1 - (ss_res_br_val / ss_tot_br_val);
    
    y_true_test_br = t(:, indTest);
    ss_res_br_test = sum((yTest_br - y_true_test_br).^2);
    ss_tot_br_test = sum((y_true_test_br - mean(y_true_test_br)).^2);
    r2_test_br(i)  = 1 - (ss_res_br_test / ss_tot_br_test);
end

% Create combined comparison plot with absolute scale
figure('Position', [100 100 800 500]);
plot(neurons, mse_train_lm, '--r', 'LineWidth', 1.5); hold on;
plot(neurons, mse_val_lm, ':r', 'LineWidth', 1.5);
plot(neurons, mse_test_lm, 'r', 'LineWidth', 2);
plot(neurons, mse_train_br, '--b', 'LineWidth', 1.5);
plot(neurons, mse_val_br, ':b', 'LineWidth', 1.5); % FIXED: Changed from ':r' to ':b'
plot(neurons, mse_test_br, 'b', 'LineWidth', 2);

% Add markers for best performance points
plot(best_neurons_lm, mse_val_lm(best_neurons_lm), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot(best_neurons_br, mse_val_br(best_neurons_br), 'bo', 'MarkerSize', 10, 'LineWidth', 2);

xlabel('Number of Neurons');
ylabel('Mean Squared Error');
title('Algorithm Performance Comparison (Normalized Input/Output)');
legend('LM Training', 'LM Validation', 'LM Testing', 'BR Training', 'BR Validation', ...
    'BR Testing', 'LM Best', 'BR Best', 'Location', 'best');
grid on;

% Display best results
fprintf('\nBest Results:\n');
fprintf('Levenberg-Marquardt (with normalized data):\n');
fprintf('  Best neurons: %d\n', best_neurons_lm);
fprintf('  Training MSE: %.6e\n', mse_train_lm(best_neurons_lm));
fprintf('  Validation MSE: %.6e\n', mse_val_lm(best_neurons_lm));
fprintf('  Testing MSE: %.6e\n', mse_test_lm(best_neurons_lm));

fprintf('\nBayesian Regularization (with normalized data):\n');
fprintf('  Best neurons: %d\n', best_neurons_br);
fprintf('  Training MSE: %.6e\n', mse_train_br(best_neurons_br));
fprintf('  Validation MSE: %.6e\n', mse_val_br(best_neurons_br));
fprintf('  Testing MSE: %.6e\n', mse_test_br(best_neurons_br));

% Create regression plots
% LM Plots
figure('Position', [100 100 1200 400]);

% Training
subplot(1,3,1);
scatter(t(:,indTrain), best_train_pred_lm, 'r');
hold on;
plot([min(t) max(t)], [min(t) max(t)], 'k--');
xlabel('True Values');
ylabel('Predicted Values');
title(sprintf('LM Training\nMSE: %.2e', mse_train_lm(best_neurons_lm)));
grid on;
axis square;

% Validation
subplot(1,3,2);
scatter(t(:,indVal), best_val_pred_lm, 'r');
hold on;
plot([min(t) max(t)], [min(t) max(t)], 'k--');
xlabel('True Values');
ylabel('Predicted Values');
title(sprintf('LM Validation\nMSE: %.2e', mse_val_lm(best_neurons_lm)));
grid on;
axis square;

% Testing
subplot(1,3,3);
scatter(t(:,indTest), best_test_pred_lm, 'r');
hold on;
plot([min(t) max(t)], [min(t) max(t)], 'k--');
xlabel('True Values');
ylabel('Predicted Values');
title(sprintf('LM Testing\nMSE: %.2e', mse_test_lm(best_neurons_lm)));
grid on;
axis square;

% BR Plots - FIXED: Corrected subplot layout
figure('Position', [100 600 1200 400]);

% Training
subplot(1,3,1);
scatter(t(:,indTrain), best_train_pred_br, 'b');
hold on;
plot([min(t) max(t)], [min(t) max(t)], 'k--');
xlabel('True Values');
ylabel('Predicted Values');
title(sprintf('BR Training\nMSE: %.2e', mse_train_br(best_neurons_br)));
grid on;
axis square;

% Validation
subplot(1,3,2);
scatter(t(:,indVal), best_val_pred_br, 'b');
hold on;
plot([min(t) max(t)], [min(t) max(t)], 'k--');
xlabel('True Values');
ylabel('Predicted Values');
title(sprintf('BR Validation\nMSE: %.2e', mse_val_br(best_neurons_br)));
grid on;
axis square;

% Testing
subplot(1,3,3);
scatter(t(:,indTest), best_test_pred_br, 'b');
hold on;
plot([min(t) max(t)], [min(t) max(t)], 'k--');
xlabel('True Values');
ylabel('Predicted Values');
title(sprintf('BR Testing\nMSE: %.2e', mse_test_br(best_neurons_br)));
grid on;
axis square;

% Choose the best algorithm based on validation error
if best_mse_br < best_mse_lm
    fprintf('\nBayesian Regularization performed better with %d neurons.\n', best_neurons_br);
    best_net = best_net_br;
else
    fprintf('\nLevenberg-Marquardt performed better with %d neurons.\n', best_neurons_lm);
    best_net = best_net_lm;
end

% Parameter importance analysis with normalized data
performanceExclusion = zeros(size(x, 1), 2);

% Loop over each input parameter
for paramIdx = 1:size(x, 1)
    % Create modified datasets excluding one parameter
    xTrainExcl = xTrain([1:paramIdx-1, paramIdx+1:end], :);
    xTestExcl = xTest([1:paramIdx-1, paramIdx+1:end], :);
    
    % Train new network with excluded parameter
    netExcl = feedforwardnet(size(best_net.layers{1}.size, 1), best_net.trainFcn);
    netExcl.divideFcn = '';
    netExcl.layers{1}.transferFcn = 'tansig';
    netExcl = configure(netExcl, xTrainExcl, tTrain);
    [netExcl, trExcl] = train(netExcl, xTrainExcl, tTrain);
    
    % Evaluate performance
    yTrainExcl_norm = netExcl(xTrainExcl);
    yTestExcl_norm = netExcl(xTestExcl);
    
    % Transform predictions back to original scale
    yTrainExcl = mapminmax('reverse', yTrainExcl_norm, ps_output);
    yTestExcl = mapminmax('reverse', yTestExcl_norm, ps_output);
    
    % Calculate MSE in original scale
    performanceExclusion(paramIdx, :) = [mean((yTrainExcl - t(:,indTrain)).^2), ...
                                        mean((yTestExcl - t(:,indTest)).^2)];
end

% Plot R^2 Comparison 
figure('Position', [100 100 800 500]);
plot(neurons, r2_train_lm, '--r', 'LineWidth', 1.5); hold on;
plot(neurons, r2_val_lm,   ':r',  'LineWidth', 1.5);
plot(neurons, r2_test_lm,  'r',   'LineWidth', 2);
plot(neurons, r2_train_br, '--b', 'LineWidth', 1.5);
plot(neurons, r2_val_br,   ':b',  'LineWidth', 1.5); % FIXED: Changed from ':r' to ':b'
plot(neurons, r2_test_br,  'b',   'LineWidth', 2);

% Markers for best R^2 (LM = best val; BR = best val)
plot(best_neurons_lm, r2_val_lm(best_neurons_lm), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot(best_neurons_br, r2_val_br(best_neurons_br), 'bo', 'MarkerSize', 10, 'LineWidth', 2);

xlabel('Number of Neurons');
ylabel('R^2');
title('Algorithm Performance Comparison (R^2)');
legend('LM Training', 'LM Validation', 'LM Testing', ...
       'BR Training', 'BR Validation', 'BR Testing', ...
       'LM Best (Val)', 'BR Best (Val)', 'Location', 'best');
grid on;

% Set y-axis limits to reasonable values for R^2
ylim([-1 1]);

% Plot RMSE Comparison 
figure('Position', [100 100 800 500]);
plot(neurons, rmse_train_lm, '--r', 'LineWidth', 1.5); hold on;
plot(neurons, rmse_val_lm,   ':r',  'LineWidth', 1.5);
plot(neurons, rmse_test_lm,  'r',   'LineWidth', 2);
plot(neurons, rmse_train_br, '--b', 'LineWidth', 1.5);
plot(neurons, rmse_val_br,   ':b',  'LineWidth', 1.5); % FIXED: Changed from ':r' to ':b'
plot(neurons, rmse_test_br,  'b',   'LineWidth', 2);

% Markers for best (lowest) RMSE
plot(best_neurons_lm, rmse_val_lm(best_neurons_lm), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot(best_neurons_br, rmse_val_br(best_neurons_br), 'bo', 'MarkerSize', 10, 'LineWidth', 2);

xlabel('Number of Neurons');
ylabel('Root Mean Squared Error');
title('Algorithm Performance Comparison (RMSE)');
legend('LM Training', 'LM Validation', 'LM Testing', ...
       'BR Training', 'BR Validation', 'BR Testing', ...
       'LM Best (Val)', 'BR Best (Val)', 'Location', 'best');
grid on;

% Calculate normalized importance
meanPerformanceImpact = mean(performanceExclusion, 2);
normalizedImpact = meanPerformanceImpact / sum(meanPerformanceImpact);

% Create parameter importance plot
figure('Position', [900 300 500 400]);
paramLabels = {'Permeability', 'Porosity', 'Layer Length'};
bar(normalizedImpact);
set(gca, 'xticklabel', paramLabels, 'XTick', 1:numel(paramLabels));
xlabel('Parameters');
ylabel('Normalized Importance');
title('Parameter Importance Analysis');
grid on;

% Add text labels above bars
for i = 1:length(normalizedImpact)
    text(i, normalizedImpact(i), sprintf('%.3f', normalizedImpact(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end