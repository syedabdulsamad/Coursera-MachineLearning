function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));



%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%

hyp =  X*theta; % basic hypothesis
sigmoidHyp = sigmoid(hyp); % sigmoid hypothesis

for i = 1:m
    first = -1 * (y(i,1) * log(sigmoidHyp(i,1)));           % cost when y = 0 
    second = (1 - y(i,1)) * log(1- sigmoidHyp(i,1));        % cost when y = 1
    J = J + (first - second);  % cost computation
end
J = J/m; % final cost

for i = 1:length(theta)  % this should be # of features
    sum = 0;
    for j = 1:m % this should be # of training examples
       sum = sum + (sigmoidHyp(j) - y(j)) * X(j,i);  
    end
    grad(i) = sum/m;  % setting gradiant simontaneously
end

% =============================================================

end
