% Get the angle between two vectors (in radians)
function [angle, axis] = getAngle(vec1, vec2)
vec1 = vec1 ./ norm(vec1);
vec2 = vec2 ./ norm(vec2);
angle = acos(dot(vec1, vec2));

if (abs(angle) < 0.0001) || (abs((angle) - pi) < 0.0001)
%     axis = [-vec1(3); vec1(2); vec1(1)];
    axis = [0; 0; 0];
    angle = 0;
%     assert(abs(acos(dot(vec1, axis)) - pi/2) < 0.001);
    return;
end

axis = cross(vec1, vec2);

% ensure the sign is correct
sa3 = sign(axis(3));
if sa3 ~= 0
    angle = angle * sa3;
end
    
end