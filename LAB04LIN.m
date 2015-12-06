%Experiment 4 - Regularization Linear Regression
clc;
close all;
x = load('ml4Linx.dat');
y = load('ml4Liny.dat');
m = length(y);

%Procedure 4.1 - Plot Data

    figure, plot(x,y,'o');
    title('Data Plot');
    x_orig = x;

    %Hypothesis has a 5th order polynomial
    x = [ones(m, 1), x, x.^2, x.^3, x.^4, x.^5];

    %Initialization of fitting parameters
    theta = zeros(6,1);
    
%Procedure 4.2   
    %Regularization parameter
    lambda = 0; %Non Regularized Linear Regression
    % Closed form solution from normal equations
    L = lambda.*eye(6); % lambda .* identity matrix 6x6
    L(1) = 0; % diagonal matrix with a zero in the upper left
    theta_L0 = ((x' * x + L)^-1) * (x' * y)
    %or theta_L0 = (x' * x + L)\x' * y %L = 6x6 zero matrix
    
    lambda = 1;
    L = lambda.*eye(6);
    L(1) = 0;
    theta_L1 = (x' * x + L)\x' * y %L = 6x6 zero matrix
    
    lambda = 10;
    L = lambda.*eye(6);
    L(1) = 0;
    theta_L10 = (x' * x + L)\x' * y 
    
%Procedure 4.3 - Plotting of the polynomial fit for each lambda
    %calculate the L2-norm of verctor x
    norm_theta_L0 = norm(theta_L0); 
    norm_theta_L1 = norm(theta_L1); 
    norm_theta_L10 = norm(theta_L10); 
    
    % Plot the linear fit for lambda = 0
    figure, plot(x_orig,y,'o');
    hold on;
    x_vals = (-1:0.025:1)';
    %Include other powers of x in feature vector x
    features = [ones(size(x_vals)), x_vals, x_vals.^2, x_vals.^3,...
              x_vals.^4, x_vals.^5];
    plot(x_vals, features*theta_L0, '--', 'LineWidth', 2)
    title('lambda = 0')
    legend('Training data', '5th order fit')
    hold off
    
    % Plot the linear fit for lambda = 1
    figure; plot(x_orig,y,'o');
    hold on;
    x_vals = (-1:0.025:1)';
    %Include other powers of x in feature vector x
    features = [ones(size(x_vals)), x_vals, x_vals.^2, x_vals.^3,...
              x_vals.^4, x_vals.^5];
    plot(x_vals, features*theta_L1, '--', 'LineWidth', 2)
    title('lambda = 1')
    legend('Training data', '5th order fit')
    hold off
    
    % Plot the linear fit for lambda = 10
    figure; plot(x_orig,y,'o');
    hold on;
    x_vals = (-1:0.025:1)';
    %Include other powers of x in feature vector x
    features = [ones(size(x_vals)), x_vals, x_vals.^2, x_vals.^3,...
              x_vals.^4, x_vals.^5];
    plot(x_vals, features*theta_L10, '--', 'LineWidth', 2)
    title('lambda = 10')
    legend('Training data', '5th order fit')
    hold off
    