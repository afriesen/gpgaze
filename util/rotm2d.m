% ----------------- Create a 2-D rotation matrix -------------------------
function R = rotm2d(gamma)
    R = [cos(gamma) -sin(gamma); sin(gamma) cos(gamma)];
end
