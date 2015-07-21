function X = trainf(x, target, iterations, X)

% train a full GP model with SE covariance function (ARD)
%
% Marc Deisenroth and Carl Edward Rasmussen, 2009-08-25

covfunc={'covSum',{'covSEard','covNoise'}};
D = size(x,2);
E = size(target,2);

if nargin < 3
  iterations = -500;
end

if ~exist('X','var') % if no initial hyper-parameters are given
  % set them to "good" heuristic values (look at statistics of the data)
  lh = repmat([log(2*std(x)) 0 -1]',1,E);
  lh(D+1,:) = log(std(target));
  lh(D+2,:) = log(std(target)/10);
else
  lh = reshape(X,D+2,E);
end

for i = 1:E % for all target dimensions
  lh(:,i) = minimize(lh(:,i),'gpr',iterations,covfunc,x,target(:,i));
end

X = reshape(lh,(D+2)*E,1); % vectorize

% check whether the hyper-parameters make sense
X = checkHypers(X, x, target);


% check hyper-parameters
function X = checkHypers(X,x,target)
D = size(x,2);
E = size(target,2);
thres = 1000;

X = reshape(X,D+2,E);
xrange = max(x)-min(x);
yrange = max(target)-min(target);
for e = 1:E
  % check length-scales
  if any(exp(X(1:D,e)) > thres.*xrange')
    disp(['target dim ' num2str(e) ': changed length-scales']);
    X(1:D,e) = min(X(1:D,e),log(thres.*xrange'));
  end
  % check signal variance
  if exp(X(D+1,e)) > 10*yrange(e)
    disp(['target dim ' num2str(e) ': changed signal variance']);
    X(D+1,e) = log(10*yrange(e));
  end
end
X = reshape(X,(D+2)*E,1);
