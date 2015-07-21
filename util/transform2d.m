function out = transform2d(in, angle, offset)
    out = zeros(size(in));
    n = size(in,3);
    
    for i = 1:n
        % rotate
        tmp = rotm2d(angle) * in(1:2,:,i); 
        
        %translate
        if any(offset ~= 0)
            tmp = [eye(2), offset; zeros(1,3)] * [tmp; ones(1, size(in,2))];
        end
        out(1:2,:,i) = tmp(1:2, :);

        % only transform the first 2 dimensions
        if rows(in) > 2
            out(:,:,i) = [out(1:2,:,i); in(3:end,:,i)];
        end
    end
end