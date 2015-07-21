function [mpi, Spi, Cxy_pi, mf, Sf, Cxy_f, sx, Ssx, sxf, Ssxf] = ...
    fwdInfGP(Xpi, xpi, ypi, Xf, xf, yf, sg, Sg, sxi, Sxi)

% inputs:
%   Xpi - hyperparams for GPpi
%   Xf  - hyperparams for GPf
%   xpi - training data for GPpi
%   ypi - training labels for GPpi
%   xf  - training data for GPf
%   yf  - training labels for Gpf
%   sg  - test data goal position
%   Sg  - covariance of the test goal position
%   sxi - test data current state
%   Sxi - covariance of the test current state

    Epi = size(ypi, 2);

    dg = size(Sg, 1);
    dxi = size(Sxi, 1);
    
    % covariance matrix for input to GP_pi
    sx = [sg; sxi];
    Ssx = [Sg, zeros(dxi, dg); zeros(dg, dxi), Sxi];
    
    % compute "predictive" distribution p(a) = \int{p(a|g)p(g)dg}
    [mpi, Spi, Cxy_pi] = gpPpi(Xpi, xpi, ypi, sx, Ssx);

    % create the covariance matrix for the input to GP_f
    sxf = [mpi; sxi];
    Ssxf = [Spi, zeros(dxi, Epi); zeros(Epi, dxi), Sxi];
        
    % compute "predictive" distribution p(x_f) = \int{p(x_f|a)p(a)da}
    [mf, Sf, Cxy_f] = gpPf(Xf, xf, yf, sxf, Ssxf);
end