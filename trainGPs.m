function [Xpi, xpi, ypi, Xf, xf, yf] = trainGPs(nobs, gazestd, gaze, ...
    goalRange, chol_S_goal, chol_S_angle, chol_S_dist, chol_S_xf, ...
    blindfold, chol_S_ang_bf, n)

noisytraining = true;

if ~exist('n', 'var'); 
    n = 200;     % number of training samples
end
niter = 1000;    % # of iterations for optimizing the hyperparams

% generate the random initial gaze positions (training)
x0 = zeros(3, n);
gaze_angles = randnorm(n, 0, gazestd); % generate w/ specified std
for i = 1:n
    x0(:, i) = qrot3d(gaze', [0; 0; 1], gaze_angles(i));
end

% % generate the angles and distances to the goal states (training)
% true_angles = angleRange(1) + (angleRange(end)-angleRange(1)).*rand(1, n); % in radians
% true_dists = distRange(1) + (distRange(end)-distRange(1)).*rand(1, n); % in centimeters

% randomly generate goals
gen_goals = [goalRange(1,1) + (goalRange(1,2)-goalRange(1,1)).*rand(1, n); ...
    goalRange(2,1) + (goalRange(2,2)-goalRange(2,1)).*rand(1, n)];
true_angles = atan2(gen_goals(2,:), gen_goals(1,:));
true_dists = sqrt(sum(gen_goals.^2));

% generate the training data
[goals, obsgoals, angles, obsangles, dists, obsdists, gazepos, xact, x, obsx] = ...
    genGazeData(nobs, x0, true_angles, true_dists, ...
    chol_S_goal, chol_S_angle, chol_S_dist, chol_S_xf, ...
    blindfold, chol_S_ang_bf);

% set up training matrices
if noisytraining
    xpi = [obsgoals; x0(1:2, :)]';
    ypi = [obsangles(:,:,1); obsdists(:,:,1)]';
else
    xpi = [goals; x0(1:2, :)]';
    ypi = [angles; dists]';
end

if noisytraining
    xf = [obsangles(:,:,1); obsdists(:,:,1); x0(1:2, :)]';
    yf = [obsx(1:2,:,1); obsdists(:,:,1)]';
else
    xf = [true_angles; true_dists; x0(1:2, :)]';
    yf = [x(1:2,:,1); true_dists]';
end

% remove x_i from the input vars if it's constant
if (gazestd == 0); xpi = xpi(:, 1:end-2); end
if (gazestd == 0); xf = xf(:, 1:end-2); end

% Train the GPs
% policy GP (trained on noise-free data)
Xpi = trainf(xpi, ypi, niter);

% transition function GP (trained on noise-free data)
Xf = trainf(xf, yf, niter);

end