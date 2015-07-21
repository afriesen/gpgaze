function [m, S, m_t, S_t, m_y S_y] = ...
  gpf(X_pi, input_pi, target_pi, X_f, input_f, target_f, pmg, pSg, xf)

% Cognitive model using GP models for policy a = pi(g) and transition
% model x_f = f(a)
% (trained offline)
% assumes that the GP models are NOT learned on differences
%
% inputs:
% X_pi:       (D+2)*E-by-1 vector of log-hyper-parameters (policy model)
% input_pi:   n-by-D matrix of training inputs (policy model)
% target_pi:  n-by-E matrix of training targets (policy model)
% X_f:        (E+2)*F-by-1 vector of log-hyper-parameters (transition model)
% input_f:    n-by-E matrix of training inputs (transition model)
% target_f:   n-by-F matrix of training targets (transition model)
% pmg:        D-by-1 mean of current (hidden) goal distribution
% pSg:        D-by-D covariance matrix of current (hidden) goal distribution
% xf:         F-by-1 measurement of next state
%
% outputs:
% m:        E-by-1 mean vector of filtered distribution
% S:        E-by-E covariance matrix of filtered distribution
% m_t:      E-by-1 mean vector of the predicted state distribution
% S_t:      E-by-E covariance matrix of the predicted state distribution
% m_y:      F-by-1 mean vector of predicted measurement distribution
% S_y:      F-by-F covariance matrix of predicted measurement distribution
% 
% Code originally by Marc Peter Deisenroth 2009-07-06
% modified by Abram L. Friesen 2011-28-01

% compute "predictive" distribution, p(a) = \int{p(a|g)p(g)}
[ma, Sa, Cga] = gpPpi(X_pi, input_pi, target_pi, pmg, pSg); % call policy GP
% [m_t S_t] = gpPt(X_t, input_t, target_t, pm, pS); 

% create the covariance matrix for the input to GP_f
% sxf = [ma(1); sx0(1:2, i)];
% if (gazestd == 0); sxf = ma(1); end
sSf = diag([Sa(1,1), 0*ones(1, Df-1)]);

% compute "predictive" distribution p(x_f) = \int{p(x_f|a)p(a)}
[mxf, Sxf, Cax] = gpPf(X_f, input_f, target_f, ma, sSf); % call transition GP
% [m_y S_y Cxy] = gpPo(X_o, input_o, target_o, m_t, S_t); % call observation GP

% filter step: combine prediction and measurement
L = chol(S_y)'; B = L\(Cxy');
m = m_t + Cxy*(S_y\(y-m_y));
S = S_t - B'*B;