% Generate gaze data for 

function [goals, obsgoals, angles, obsangles, dists, obsdists, gazepos, x_act, x, obsx] = ...
    genGazeData(nobs, x0, angles_in, dists_in, ch_S_goal, ch_S_angle, ch_S_dist, ch_S_xf, ...
    blindfold, ch_S_angbf)

% Generate simulated gaze-following data.
%
% Abe Friesen Jan 20, 2011
%
% inputs:
%   nobs        number of observations of final position to take
%   x0          initial agent gaze (position is assumed to be [0;0;0]
%   angles_in   angles from the initial state to find the actual targets
%   dists_in    distances from the initial state to find the targets
%   ch_S_goal   cholesky decomp. of covariance matrix for the noisy goal state
%   ch_S_angle  cholesky decomp. of covariance for the noisy angle
%   ch_S_dist   cholesky decomp. of covariance for noisy distance measurements
%   ch_S_xf     cholesky decomp. of cov for final gaze vector (angular noise)
%   blindfold   true if the agent is blindfolded
%   ch_S_angbf  cholesky decomp. of cov. for noisy angle with blindfold
%
% outputs:
%   goals       actual goal states (positions to gaze to)
%   obsgoals    observed goal states (sampled from N(goals, S_goal))
%   angles      actual angles to the observed goal states
%   obsangles   observed angles to the observed goal states
%   dists       actual distances to the observed goal states
%   obsdists    observed distances to the observed goal states
%   gazepos     position that is actually being looked at (with all noise)
%   x_act       actual agent state to look at the goal
%   x           agent state for looking at gazepos
%   obsx        observed agent state for looking at gazepos (x + noise)


n = length(angles_in);
assert(length(angles_in) == length(dists_in));

rot_vec = [0; 0; 1];
if (size(x0, 2) ~= n); x0 = repmat(x0, 1, n); end

if nargin < 9; blindfold = false; end

% allocate space
% goals =     zeros(3, n);
angles =    zeros(1, n);
dists =     zeros(1, n);
% obsgoals =  zeros(3, n);
obsangles = zeros(1, n, nobs);
obsdists =  zeros(1, n, nobs);
gazepos =   zeros(3, n, nobs);
% x_act =     zeros(3, n);
x =         zeros(3, n, nobs);
obsx =      zeros(3, n, nobs);

% determine the actual (true) goal position (position to gaze to)
% for i = 1:n
%     goals(:, i) = dists_in(i) .* qrot3d(x0(:, i)', rot_vec, angles_in(i))';
% end
goals = [repmat(dists_in, 2, 1) .* [cos(angles_in); sin(angles_in)]; zeros(1,n)];

% sample the noisy goals
obsgoals = randnorm(n, goals, ch_S_goal);

% get the angles and distances to the noisy goals
if ~blindfold
    for i = 1:n
        angles(i) = getAngle(x0(:, i), obsgoals(:, i));
        dists(i) = norm(obsgoals(:, i));
    end
else
    angles = randnorm(n, 0, ch_S_angbf);
    dists = randnorm(n, mean(dists_in), 1.25*std(dists_in));
end

% sample the noisy angles and distances
for j = 1:nobs
    obsangles(:, :, j) = randnorm(n, angles, ch_S_angle);
    obsdists(:, :, j) = randnorm(n, dists, ch_S_dist);
end

x_act = [cos(angles_in); sin(angles_in)];

% get the final gaze positions and agent states
for i = 1:n
%     x_act(:, i) = qrot3d(x0(:, i)', rot_vec, angles_in(i))';
    
    for j = 1:nobs
        gazepos(:,i,j) = obsdists(1,i,j) .* qrot3d(x0(:, i)', rot_vec, obsangles(1,i,j))';
        x(:, i, j) = qrot3d(x0(:, i)', rot_vec, obsangles(1,i,j))';
        
        a = obsangles(1,i,j) + randn(1,1)*ch_S_xf;
        obsx(:, i, j) = qrot3d(x0(:, i)', rot_vec, a)';
    end
end

goals = goals(1:2, :);
obsgoals = obsgoals(1:2, :);
gazepos = gazepos(1:2, :, :);
x_act = x_act(1:2, :);
x = x(1:2, :, :);
obsx = obsx(1:2, :, :);
end
