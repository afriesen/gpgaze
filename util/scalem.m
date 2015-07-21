% ----------------- Create a scaling matrix ------------------------------
function S = scalem(s,dim)
    if nargin < 2; dim = 3; end
    S = s*eye(dim);
end