% ----------------- Plot the contours of a 2-D Gaussian ------------------
function plotGaussContour(mu, sigma, dims, range, step)
    [X,Y] = meshgrid(range(1,1):step:range(1,2), ...
        range(2,1):step:range(2,2));
    
    if size(mu,2) == 1; mu = mu'; end
    mu2 = mu(dims);
    sigma2 = sigma(dims,dims);

    try
        Z = mvnpdf1([reshape(X,[],1), reshape(Y,[],1)], mu2, sigma2);
        Z = reshape(Z, size(X));

        contour(X,Y,Z);
    catch me
        fprintf('Plotting gaussian contours failed\n');
    end
end