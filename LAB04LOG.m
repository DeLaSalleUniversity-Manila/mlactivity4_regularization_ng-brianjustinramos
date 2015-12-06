%Experiment 4 - Regularization Logistic Regression
clc;
close all;
x = load('titan.dat');
y = load('surv.dat');
m = length(y);

%Procedure 4.4 - Plot Data

    figure
    pos = find(y);
    neg = find(y==0);
    plot(x(pos,1),x(pos,2),'+')
    hold on
    plot(x(neg,1),x(neg,2),'o')
    title('Data Plot')
    xlabel('Age')
    ylabel('Fare')
    
    figure
    pos = find(y);
    neg = find(y==0);
    plot(x(pos,1),x(pos,2),'+')
    hold on
    plot(x(neg,1),x(neg,2),'o')
    
    %Use map_feature.m file - maps the original inputs to the feature vector.
    x = map_feature(x(:,1), x(:,2));
    [m, n] = size(x);
    theta = zeros(n, 1); %Initialization of fitting parameters

%Procedure 4.5 - Newton's Method using values of lambda    
    %Sigmoid Function
    g = inline('1.0 ./ (1.0 + exp(-z))');
    
    MAX_ITR = 15;
    J = zeros(MAX_ITR, 1);
    
    lambda = 0;
    for i = 1:MAX_ITR
        % Calculate the hypothesis function
        z = x * theta;
        h = g(z);

        % Logistic Regression Cost Function with a regularization term
        % Cost Function: J(i) =(1/m)*sum(-y.*log(h) - (1-y).*log(1-h))
        % Regularization term: lambda/(2*m))*norm(theta([2:end]))^2;
        J(i) =(1/m)*sum(-y.*log(h) - (1-y).*log(1-h))+ ...
        (lambda/(2*m))*norm(theta([2:end]))^2;

        % Calculate gradient and hessian.
        G = (lambda/m).*theta; G(1) = 0; % extra term for gradient
        L = (lambda/m).*eye(n); L(1) = 0;% extra term for Hessian
        grad = ((1/m).*x' * (h-y)) + G; %Gradient with extra term
        H = ((1/m).*x' * diag(h) * diag(1-h) * x) + L; %Hessian with extra term

        % Using update rule
        theta = theta - H\grad;
        %ite = i

    end

%Procedure 4.6 - Print Value of J    
    J

%Proecdure 4.7 - Plotting of Decision Boundary    
    norm_theta = norm(theta) 
    
    % Define the ranges of the grid
    u = linspace(-1, 1.5, 200);
    v = linspace(-1, 1.5, 200);
    % Initialize space for the values to be plotted
    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = map_feature(u(i), v(j))*theta;
        end
    end
    % Because of the way that contour plotting works
    % in Matlab, we need to transpose z, or
    % else the axis orientation will be flipped!
    z = z';
    % Plot z = 0 by specifying the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
    legend('y = 1', 'y = 0', 'Decision boundary')
    title(sprintf('\\lambda = %g', lambda), 'FontSize', 14)
    
    hold off
    