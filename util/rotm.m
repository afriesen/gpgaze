% ----------------- Create a 3-D rotation matrix -------------------------
function R = rotm(alpha, beta, gamma)
    R = [cosd(gamma) -sind(gamma) 0; sind(gamma) cosd(gamma) 0; 0 0 1] * ... rotate around z
        [cosd(beta) 0 sind(beta); 0 1 0; -sind(beta) 0 cosd(beta)] * ... rotate around y
        [1 0 0; 0 cosd(alpha) -sind(alpha); 0 sind(alpha) cosd(alpha)]; % rotate around x
end