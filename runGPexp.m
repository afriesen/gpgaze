% note: we assume that trainGazeGP has already been run so we assume that
% the necessary parameters and data are available.

retrainGPs = false;

ntt = 4;        % number of diff trials
nGP = 12;       % number of GPs == number of "children"
n = 30;         % number of training data points to use
nblind = nGP / 3; % # GPs with blindfold experience
assert(mod(nGP,3) == 0); % make sure nblind is an integer 
assert(mod(nGP,ntt) == 0); % make sure we get equal numbers of trial types
isblind = logical([ones(1, nblind), zeros(1, nblind), zeros(1, nblind)]);
isctrl = logical([zeros(1, nblind), ones(1, nblind), zeros(1, nblind)]);

nobs = 10;

% mentorpos = [600; 0; 0];

% these are the ranges for the goal fixation points for looking left and right
leftGoalRange = [250, 350; 250, 350];
rightGoalRange = diag([1 -1]) * leftGoalRange;

% these are the possible test example orders (Left is -1, Right is +1)
order1 = [-1, 1, -1, 1]; order2 = [-1, 1, 1, -1];
order3 = -order1; order4 = -order2;
orders = [order1; order2; order3; order4];

% randomly choose test trial orders but make sure that there's an equal
% number of each
trialordersbf = mod(randperm(nblind), ntt) + 1;
trialorders = mod(randperm(nblind), ntt) + 1;
trialordersctrl = mod(randperm(nblind), ntt) + 1;


%% ------------------------- train the GPs --------------------------------
if ~exist('Xpi', 'var') || ~isequal(size(Xpi), [6*Epi, nGP]) || retrainGPs

Xpi = zeros(6*Epi, nGP); Xf = zeros(6*Ef, nGP);
xpi = zeros(n, Dpi, nGP); ypi = zeros(n, Epi, nGP);
xf = zeros(n, Df, nGP); yf = zeros(n, Ef, nGP);
Xpibf = zeros(6*Epi, nblind);
xpibf = zeros(n, Dpi, nblind); ypibf = zeros(n, Epi, nblind);

% train each GP
fprintf('training GPs\n');
for i = 1:nGP
    [Xpi(:,i), xpi(:,:,i), ypi(:,:,i), Xf(:,i), xf(:,:,i), yf(:,:,i)] = ...
        trainGPs(nobs, gazestd, gaze, goalRange, chol_S_goal, chol_S_angle, ...
            chol_S_dist, chol_S_xf, false, chol_S_ang_bf, n);
        
    % train some of the "children" with the blindfold
    if i <= nblind
        [Xpibf(:,i), xpibf(:,:,i), ypibf(:,:,i)] = ...
            trainGPs(nobs, gazestd, gaze, goalRange, chol_S_goal, chol_S_angle, ...
                chol_S_dist, chol_S_xf, true, chol_S_ang_bf, n);
    end
end
end


%% ----------------------- generate test data -----------------------------

sx0 = gaze;
ntest = ntt * nGP;

% generate a grid of goals for looking left
gengoals = [leftGoalRange(1,1) + (leftGoalRange(1,2)-leftGoalRange(1,1)).*rand(1, ntest/2); ...
    leftGoalRange(2,1) + (leftGoalRange(2,2)-leftGoalRange(2,1)).*rand(1, ntest/2)];
lxsa = atan2(gengoals(2,:), gengoals(1,:));
lxsd = sqrt(sum(gengoals.^2));

[lsg, lsog, lsa, lsoa, lsd, lsod, lsgp, lsxact, lsx, lsox] = ...
    genGazeData(nobs, sx0, lxsa, lxsd, chol_S_goal, chol_S_angle, chol_S_dist, chol_S_xf);

% generate a grid of goals for looking right
gengoals = [rightGoalRange(1,1) + (rightGoalRange(1,2)-rightGoalRange(1,1)).*rand(1, ntest/2); ...
    rightGoalRange(2,1) + (rightGoalRange(2,2)-rightGoalRange(2,1)).*rand(1, ntest/2)];
rxsa = atan2(gengoals(2,:), gengoals(1,:));
rxsd = sqrt(sum(gengoals.^2));

[rsg, rsog, rsa, rsoa, rsd, rsod, rsgp, rsxact, rsx, rsox] = ...
    genGazeData(nobs, sx0, rxsa, rxsd, chol_S_goal, chol_S_angle, chol_S_dist, chol_S_xf);

% transform into agent coordinates
tlsg = tfmtoa(lsg); tlsog = tfmtoa(lsog); tlsgp = tfmtoa(lsgp); 
talsgp = atan2(tlsgp(2,:,:), tlsgp(1,:,:)); tlsx = [cos(talsgp); sin(talsgp)];
tlsox = tfmtoa(repmat(lsod,2,1).*lsox); tlsox = atan2(tlsox(2,:,:), tlsox(1,:,:)); 
tlsox = [cos(tlsox); sin(tlsox)];

trsg = tfmtoa(rsg); trsog = tfmtoa(rsog); trsgp = tfmtoa(rsgp);
tarsgp = atan2(trsgp(2,:,:), trsgp(1,:,:)); trsx = [cos(tarsgp); sin(tarsgp)];
trsox = tfmtoa(repmat(rsod,2,1).*rsox); trsox = atan2(trsox(2,:,:), trsox(1,:,:)); 
trsox = [cos(trsox); sin(trsox)];

%% ---------------------------- run tests ---------------------------------

% space for reverse inference results
maf = zeros(Epi, nGP, ntt);
Saf = zeros(Epi, Epi, nGP, ntt);
mgf = zeros(Dpi, nGP, ntt);
Sgf = zeros(Dpi, Dpi, nGP, ntt);

% space for forward inference results
amgf = zeros(Dpi, nGP, ntt);
aSgf = zeros(size(Sgf));
m_pi = zeros(Epi, nGP, ntt);
S_pi = zeros(Epi, Epi, nGP, ntt);
m_f = zeros(Ef, nGP, ntt);
S_f = zeros(Ef, Ef, nGP, ntt);

sides = zeros(nGP, ntt); % the side that is looked to on each trial
testidx = zeros(nGP, ntt); % the index into the side that is looked on

% sg, sog, sa, soa, sd, sod, sgp, sxact, sx, sox

for no = 1:nobs

tidx = 1; tidxctrl = 1;tidxbf = 1; % trial indices (no blindfold exp., windowed control, blindfold exp.)
lidx = 1; ridx = 1;   % left/right side indices to keep track of which test points we've used

for i = 1:nGP
for j = 1:ntt
        
    if no == 1
        % use the default prior to start
        s_g = m_g_prior; S_g = diag(S_g_prior_rev);
    else
        % use the previously computed value as the new prior
        s_g = mgf(1:2, i, j); S_g = Sgf(1:2, 1:2, i, j);
    end

    % observed x_i and x_f
    sxi = sx0(1:2); Sxi = diag(S_xi_prior); % mentor always starts looking straight ahead
        
    % determine the side that the test data is drawn from
    if isblind(i)
        sides(i,j) = orders(trialordersbf(tidxbf), j);
    elseif isctrl(i)
        sides(i,j) = orders(trialordersctrl(tidxctrl), j);
    else
        sides(i,j) = orders(trialorders(tidx), j);
    end
    
    if sides(i,j) > 0 % right
        xf_obs = [rsox(1:2,ridx,no); rsod(1,ridx,no)];
        testidx(i,j) = ridx;
        ridx = ridx + 1;
    else  % left
        xf_obs = [lsox(1:2,lidx,no); rsod(1,lidx,no)];
        testidx(i,j) = lidx;
        lidx = lidx + 1;
    end
    
    % reverse inference
    if isblind(i)    
        [maf(:,i,j), Saf(:,:,i,j), mgf(:,i,j), Sgf(:,:,i,j)] = ...
            revInfGP(Xpibf(:,i), xpibf(:,:,i), ypibf(:,:,i), Xf(:,i), xf(:,:,i), yf(:,:,i), ...
                s_g, S_g, sxi, Sxi, xf_obs, nsamples);
    else
        [maf(:,i,j), Saf(:,:,i,j), mgf(:,i,j), Sgf(:,:,i,j)] = ...
            revInfGP(Xpi(:,i), xpi(:,:,i), ypi(:,:,i), Xf(:,i), xf(:,:,i), yf(:,:,i), ...
                s_g, S_g, sxi, Sxi, xf_obs, nsamples);
    end

    % convert from the mentor's coordinate frame to the agent's
    amgf(:, i, j) = tfmtoa(mgf(:, i, j));
    aSgf(1:2,1:2,i,j) = rotm2d(rot_angle)*Sgf(1:2,1:2,i,j)*rotm2d(rot_angle)';

    % set up the inputs for the forward inference
    s_g = amgf(1:2, i, j); S_g = aSgf(1:2, 1:2, i, j);
    sxi = gaze(1:2); Sxi = zeros(2);    
    if gazestd == 0;
        sxi = zeros(0, ntest); Sxi = zeros(0);
    end

    % forward inference
    [m_pi(:,i,j), S_pi(:,:,i,j), ~, m_f(:,i,j), S_f(:,:,i,j), ~] = ...
        fwdInfGP(Xpi(:,i), xpi(:,:,i), ypi(:,:,i), Xf(:,i), xf(:,:,i), yf(:,:,i), ...
            s_g, S_g, sxi, Sxi);
        
end % ntt

if isblind(i)
    tidxbf = tidxbf + 1; 
elseif isctrl(i)
    tidxctrl = tidxctrl + 1;
else
    tidx = tidx + 1;
end

end % nGP

%% ---------------------------- plot -------------------------------------

figure(1); clf; hold on;
rectangle('Position', [(agentpos(1:2)' - [1 1]) 2 2]);
% rectangle('Position', [(mentorpos(1:2)' - [1 1]) 2 2]);
title('gaze vectors (reverse + forward) - blindfold experiment');
for i = 1:nGP
    for j = 1:ntt
        ti = testidx(i,j);
        s = 'r+';
        if isblind(i); s = 'bo'; end
        if sides(i, j) > 0 % right
%             plot([ntsog(1,:); m_f(1,i,j)], ...
%                 [ntsog(2,:); m_f(2,i,j)], 'y');
            plot([mean(trsox(1,ti,:),3); m_f(1,i,j)], ...
                [mean(trsox(2,ti,:),3); m_f(2,i,j)], 'c');
            plot(trsx(1, ti), trsx(2, ti), 'g. ');
            plot(mean(trsox(1,ti,:),3), mean(trsox(2,ti,:),3), 'b+ ');
%             plot(ntsog(1, :), ntsog(2, :), 'ko ');
            plot(m_f(1,i,j), m_f(2,i,j), s);
        else % left
%             plot([ntsog(1,:); m_f(1,i,j)], ...
%                 [ntsog(2,:); m_f(2,i,j)], 'y');
            plot([mean(tlsox(1,ti,:),3); m_f(1,i,j)], ...
                [mean(tlsox(2,ti,:),3); m_f(2,i,j)], 'c');
            plot(tlsx(1, ti), tlsx(2, ti), 'g. ');
            plot(mean(tlsox(1,ti,:),3), mean(tlsox(2,ti,:),3), 'b+ ');
%             plot(ntsog(1, :), ntsog(2, :), 'ko ');
            plot(m_f(1,i,j), m_f(2,i,j), s);
        end
    end
end
axis square; hold off;

ap = agentpos(1:2)';
mp = mentorpos(1:2)';

figure(2); clf; hold on;
boxsize = [80 80];
rectangle('Position', [(ap - boxsize/2) boxsize], 'facecolor', 'r', 'Curvature', [1 1]);
rectangle('Position', [(mp - boxsize/2) boxsize], 'facecolor', 'b', 'Curvature', [1 1]);
% arrow fixlimits;
% title('goal positions (reverse + forward) - blindfold experiment');
for i = 1:nGP
    for j = 1:ntt
        ti = testidx(i,j);
        s = 'r+';
%         if isblind(i); s = 'ko'; end
        if isblind(i); continue; end;
        if sides(i, j) > 0 % right
% %             plot([ntsog(1,:); m_f(1,i,j)], ...
% %                 [ntsog(2,:); m_f(2,i,j)], 'y');
%             plot([mean(trsog(1,ti,:),3); m_f(3,i,j)*m_f(1,i,j)], ...
%                 [mean(trsog(2,ti,:),3); m_f(3,i,j)*m_f(2,i,j)], 'c');
            plot(trsg(1, ti), trsg(2, ti), 'bx ');
%             plot(mean(trsog(1,ti,:),3), mean(trsog(2,ti,:),3), 'b+ ');
% %             plot(ntsog(1, :), ntsog(2, :), 'ko ');
            plot(m_f(3,i,j)*m_f(1,i,j), m_f(3,i,j)*m_f(2,i,j), s);
        else % left
% %             plot([ntsog(1,:); m_f(1,i,j)], ...
% %                 [ntsog(2,:); m_f(2,i,j)], 'y');
%             plot([mean(tlsog(1,ti,:),3); m_f(3,i,j)*m_f(1,i,j)], ...
%                 [mean(tlsog(2,ti,:),3); m_f(3,i,j)*m_f(2,i,j)], 'c');
            plot(tlsg(1, ti), tlsg(2, ti), 'bx ');
%             plot(mean(tlsog(1,ti,:),3), mean(tlsog(2,ti,:),3), 'b+ ');
% %             plot(ntsog(1, :), ntsog(2, :), 'ko ');
            plot(m_f(3,i,j)*m_f(1,i,j), m_f(3,i,j)*m_f(2,i,j), s);
        end
    end
end
% axis square; 
axis([-150 700 -600 400])
arrow(ap, ap+[75,0], 10, 60, 30);
arrow(mp, mp-[75,0], 10, 60, 30);
i = 10; j = 1; scale = 0.75;
ti = testidx(i, j); side = sides(i, j);
arrow(mp, mp + scale*(trsg(:,ti)' - mp), 10, 60, 30);
arrow(ap, scale*m_f(3,i,j)*[m_f(1,i,j); m_f(2,i,j)], 10, 60, 30);
leg = legend('true fixation points', 'inferred fixation points');
text(-70, -100, 'Agent');
text(mp(1) - 80, mp(2) - 100, 'Mentor');

grid off;

xlabel('x position (cm)');
ylabel('y position (cm)');

% leg = legend(filternames{i},'true');
set(leg, 'box', 'on', 'FontSize', 26, 'location', 'southwest');
hold off;

% --------------------------- score model --------------------------------
scores = zeros(nGP, ntt);
minangle = 35;

amf = squeeze(atan2(m_f(2,:,:), m_f(1,:,:)) * 180/pi);

for i = 1:nGP
    for j = 1:ntt
        if sides(i, j) > 0 % right
            if amf(i,j) <= -minangle
                scores(i, j) = -1;
            elseif amf(i,j) >= minangle
                scores(i, j) = 1;
            end
        else % left
            if amf(i,j) >= minangle
                scores(i, j) = -1;
            elseif amf(i,j) <= -minangle
                scores(i,j) = 1;
            end
        end
    end
end

gp_score = sum(scores, 2);
gps = gp_score(~isblind & ~isctrl);
gpsctrl = gp_score(isctrl);
gpsbf = gp_score(isblind);

m = mean([gps, gpsctrl, gpsbf])';
sd = std([gps, gpsctrl, gpsbf])' / sqrt(nblind);
%%
figure(3); clf;
h = barweb(m, sd, 0.6, {'Baseline'; 'Window'; 'Opaque'}, [], [], 'Looking Score', [], [], [], 1);
set(h.bars(1), 'FaceColor', 'w')

% ch = get(h.bars, 'children');
% fvd = get(ch, 'Faces');
% fvcd = get(ch, 'FaceVertexCData');
% fvcd(fvd(2,:)) = 1;
% set(ch, 'FaceVertexCData', fvcd);

set(gca, 'YTick', [0 1 2 3 4]);
 	ylim([0 4]);
% 	xlim([0.5 numgroups-change_axis+0.5]);
% print_fig('bf_gaze_exp_bar')
drawnow;

end % nobs
