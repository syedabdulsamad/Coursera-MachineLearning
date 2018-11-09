function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



hyp =  X*theta; % basic hypothesis
sigmoidHyp = sigmoid(hyp); % sigmoid hypothesis
tempCost = 0;
for i = 1:m
    first = -1 * (y(i,1) * log(sigmoidHyp(i,1)));           % cost when y = 0 
    second = (1 - y(i,1)) * log(1- sigmoidHyp(i,1));        % cost when y = 1
    tempCost = tempCost + (first - second);  % cost computation
end
tempCost = tempCost/m;

regularizedTerm = 0;
for t = 2:length(theta)
   regularizedTerm = regularizedTerm + theta(t)^2;
end

tempCost2 = (lambda/(2*m)) * regularizedTerm;
J = tempCost + tempCost2; % final cost

%-------------------------------Gradiant Calculation----------------------------------%

% Gradiant calculation for theta 0
firstSum = 0;
for j = 1:m % this should be # of training examples
    firstSum = firstSum + (sigmoidHyp(j) - y(j)) * X(j,1);      
end 
grad(1) = firstSum/m;

% Gradiant calculation for remaining theta
for i = 2:length(theta)  % this should be # of features
    sum = 0;
    for j = 1:m % this should be # of training examples
       sum = sum + (sigmoidHyp(j) - y(j)) * X(j,i);  
    end
    grad(i) = (sum/m + (lambda/m * theta(i)));  % setting gradiant simontaneously
end

end
