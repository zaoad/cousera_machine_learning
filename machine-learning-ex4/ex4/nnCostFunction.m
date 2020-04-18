function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
j=0;
first_X=[ones(m,1) X];
second_layer=sigmoid(first_X*Theta1');
second_a=[ones(m,1) second_layer];
final_layer=sigmoid(second_a*Theta2');
h_theta=final_layer;
temp_check=sum(sum(h_theta));
%fprintf("\n check htheta %f\n", temp_check);
y_temp=zeros(m, num_labels);
for i=1:m
 y_temp(i,y(i))=1;
end
cost_value=(-1*y_temp).*log(h_theta)-(1-1*y_temp).*log(1-h_theta);
J=(1/m)*sum(sum(cost_value));
t1_temp=Theta1;
t1_temp(:,1)=0;
t2_temp=Theta2;
t2_temp(:,1)=0;
reg_cost=(lambda/(2*m))*(sum(sum(t1_temp.^2))+sum(sum(t2_temp.^2)))
%fprintf("value of j %f \n and value of Reg %f\n", j , reg_cost);
J=J+reg_cost;


%backpropagation
for t=1:m
  a_1=X(t,:);
  a_1=[1 a_1];
  %disp(size(a_1));
  z_2=a_1*Theta1';
  %disp(size(z_2));
  a_2=sigmoid(z_2);%(1*25)
  a_2=[1 a_2];
  %disp(size(a_2));
  z_3=a_2*Theta2';
  %disp(size(z_3));
  a_3=sigmoid(z_3);
  delta_3=a_3-y_temp(t,:);
  z_2=[1 z_2];
  delta_2=(delta_3*Theta2).* sigmoidGradient(z_2);%(1*26)
  delta_2=delta_2(1, 2:end);%(1*25)
  %disp(size(delta_2));
  Theta2_grad=Theta2_grad+delta_3'*a_2;
  Theta1_grad=Theta1_grad+delta_2'*a_1;
  
end;

Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)




Theta1_grad(: , 2:end)=Theta1_grad(: , 2:end)+(lambda/m)*Theta1(: , 2:end);

Theta2_grad(: , 2:end)=Theta2_grad(: , 2:end)+(lambda/m)*Theta2(: , 2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
