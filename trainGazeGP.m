run gpml-matlab-v3.1/startup.m;
addpath( 'qrot3d' );  % qrot3d may need to be mex'd before running this
addpath( 'gpadf' );
addpath( 'util' );

% some defaults for the plots
set(0,'defaultaxesfontsize',28);
set(0,'defaultaxesfontunits', 'points')
set(0,'defaulttextfontsize',32);
set(0,'defaulttextfontunits','points')
set(0,'defaultaxeslinewidth',1);
set(0,'defaultlinelinewidth',2);
set(0,'DefaultAxesLineStyleOrder','-|--|:|-.');
set(0,'DefaultLineMarkerSize',13);

covfunc={'covSum',{'covSEard','covNoise'}};
dotraining = false;
testfwdinf = false;

noisescalar = 1;

% agent state
agentpos = [0; 0; 0];
gaze = [1; 0; 0]; gazestd = 30 * pi/180;

% mentor state (agent is assumed to be at [0; 0] with gaze [1; 0])
mentorpos = [600; 0; 0];
mentorgaze = [-1; 0; 0];

goalRange = [100, 500; -500, 500];
testGoalRange = [200, 400; -400, 400];

% angleRange = [-90, 90] * pi/180; % radians
% distRange = [100, 600];          % cm
% testAngleRange = [-70, 70] *pi/180; % radians
% testDistRange = [150, 250];

nobs = 5;        % number of observations to take and use
nsamples = 100;  % number of samples to use of the inverse inference for GP_pi() 

% na = 10; nd = 5; ntest = na*nd; % # test data points
nx_test = 5; ny_test = 10; ntest = nx_test*ny_test;

blindfold = false;
chol_S_ang_bf = 40 *pi/180;

chol_S_goal = diag([10; 10; 0]) * noisescalar;
chol_S_angle = (3 * pi/180) * noisescalar;
chol_S_dist = 10 * noisescalar;
chol_S_xf = 3 * pi/180 * noisescalar; % noise alters the angle that xf is at since xf is a gaze vector

m_g_prior = [mean(testGoalRange(2,:)); 0];
% m_g_prior = [-100; 0; 0];
S_g_prior_fwd = diag([0; 0]) * noisescalar;
S_g_prior_rev = 1000*[80; 20];

m_xi_prior = [1; 0];
S_xi_prior = [1; 1] * pi/180;
if (gazestd == 0)
    m_xi_prior = []; 
    S_xi_prior = [];
end


%% --------------------- train the GPs ------------------------------------

if ~exist('Xpi', 'var') || (size(Xpi,2) ~= 1) || dotraining
    [Xpi, xpi, ypi, Xf, xf, yf] = trainGPs(nobs, gazestd, gaze, ....
        goalRange, chol_S_goal, chol_S_angle, chol_S_dist, ...
        chol_S_xf, blindfold, chol_S_ang_bf);
end

Dpi = size(xpi, 2);
Epi = size(ypi, 2);
Df = size(xf, 2);
Ef = size(yf, 2);

%% ------------------- create test data -----------------------------------

sx0 = zeros(3, ntest);
sgz_ang = randnorm(ntest, 0, gazestd);
for i = 1:ntest
    sx0(:, i) = qrot3d(gaze', [0; 0; 1], sgz_ang(i));
end

% generate a grid of goals
[ggX, ggY] = meshgrid(linspace(testGoalRange(1,1), testGoalRange(1,2), nx_test), ...
    linspace(testGoalRange(2,1), testGoalRange(2,2), ny_test));

xsa = atan2(ggY, ggX);
xsd = sqrt(ggX.^2 + ggY.^2);

% [xsa, xsd] = meshgrid(linspace(testAngleRange(1), testAngleRange(end), na), ...
%     linspace(testDistRange(1), testDistRange(2), nd));
xsa = reshape(xsa, 1, []);
xsd = reshape(xsd, 1, []);

% generate the test data
[sg, sog, sa, soa, sd, sod, sgp, sxact, sx, sox] = ...
    genGazeData(nobs, sx0, xsa, xsd, ...
    chol_S_goal, chol_S_angle, chol_S_dist, chol_S_xf, ...
    blindfold, chol_S_ang_bf);

% transform the test data into the agent's coordinate frame (we assume that
% it is generated in the mentor's)
gaze_angle = atan2(mentorpos(2), mentorpos(1)); % have the agent always looking at the mentor
% gaze_angle = atan2(gaze(2), gaze(1));
mentor_gaze_angle = atan2(mentorgaze(2), mentorgaze(1));
rot_angle = mentor_gaze_angle - gaze_angle;
trans_dist = (mentorpos - agentpos);
tsx0 = repmat([cos(gaze_angle); sin(gaze_angle)], 1, ntest);

% transformation functions
tfmtoa = @(in) (transform2d(in, rot_angle, trans_dist(1:2)));
tfatom = @(in) (transform2d(in, -rot_angle, -trans_dist(1:2)));

tsg = tfmtoa(sg); tsog = tfmtoa(sog); tsgp = tfmtoa(sgp);
% tsa = atan2(tsog(2,:), tsog(1,:));
tsagp = atan2(tsgp(2,:,:), tsgp(1,:,:));
tsx = [cos(tsagp); sin(tsagp)];
tsox = tfmtoa(repmat(sod,2,1).*sox); tsox = atan2(tsox(2,:,:), tsox(1,:,:)); 
tsox = [cos(tsox); sin(tsox)];

% pre-compute quantites for the sampled observed goal (angle, dist, gaze vector)
asg = atan2(sg(2,:), sg(1,:));
asog = atan2(sog(2, :), sog(1,:));
asx0 = atan2(sx0(2,:), sx0(1,:));
aasog = asog - asx0;
dsog = sqrt(sum(sog.^2, 1));
nsog = bsxfun(@rdivide, sog, dsog);

atsg = atan2(tsg(2,:), tsg(1,:));
atsog = atan2(tsog(2,:), tsog(1,:));
atsx0 = atan2(tsx0(2,:), tsx0(1,:));
aatsog = atsog - atsx0;
dtsog = sqrt(sum(tsog.^2, 1));
ntsog = bsxfun(@rdivide, tsog, dtsog);


%% ----------------- test forward inference -------------------------------

% if ~exist('fwd_m_pi', 'var') || testfwdinf
fwd_m_pi = zeros(ntest, Epi);
fwd_S_pi = zeros(ntest, Epi, Epi);
fwd_Cga_pi = zeros(ntest, Dpi, Epi);

fwd_m_f = zeros(ntest, Ef);
fwd_S_f = zeros(ntest, Ef, Ef);
fwd_Cax_f = zeros(ntest, Df, Ef);

fwd_sSpi = diag([diag(S_g_prior_fwd).^2; 0*ones(Dpi-length(S_g_prior_fwd), 1)]);
Sg_fwd = S_g_prior_fwd;

sxi = sx0(1:2, :); Sxi = zeros(2);
if gazestd == 0;
    sxi = zeros(0, ntest); Sxi = zeros(0);
end

for i = 1:ntest
    % Evaluate the GPs in the forward direction
    %  compute "predictive" distribution p(a) = \int{p(a|g)p(g)dg} = N(ma,Sa)
    %  compute "predictive" distribution p(x_f) = \int{p(x_f|a)p(a)da} = N(mxf,Sxf)
    [fwd_m_pi(i,:), fwd_S_pi(i,:,:), fwd_Cga_pi(i,:,:), ...
        fwd_m_f(i,:), fwd_S_f(i,:,:), fwd_Cax_f(i,:,:)] = ...
        fwdInfGP(Xpi, xpi, ypi, Xf, xf, yf, sog(:, i), Sg_fwd, sxi(:,i), Sxi);
end
% end

%% ------------------- reverse inference ----------------------------------

maf = zeros(Epi, ntest);
Saf = zeros(Epi, Epi, ntest);
mgf = zeros(Dpi, ntest);
Sgf = zeros(Dpi, Dpi, ntest);

% use multiple samples of the observation to get more accurate results (by
% overwriting the prior)
for no = 1:nobs
fprintf('starting reverse inference with observation %d\n', no);
    
for i = 1:ntest

    if no == 1
        % use the default prior to start
        s_g = m_g_prior; S_g = diag(S_g_prior_rev);
    else
        % use the previously computed value as the new prior
        s_g = mgf(1:2, i); S_g = Sgf(1:2, 1:2, i);
    end
    
    sxi = sx0(1:2, i); % observed x_i
    Sxi = diag(S_xi_prior);
    
    xf_obs = [sox(1:2,i,no); sod(1,i,no)]; % measured value for x_f

    % infer backwards in our model
    [maf(:, i), Saf(:,:,i), mgf(:, i), Sgf(:,:,i)] = ...
        revInfGP(Xpi, xpi, ypi, Xf, xf, yf, s_g, S_g, sxi, Sxi, xf_obs, nsamples);
    
end

% mean(Sgf, 3)


%% --------------------- forward inference --------------------------------
% infer the gaze vector (x_f) given the mentor's goal (g) in order to 
% look where the mentor was looking

fprintf('starting forward inference with observation %d\n', no);

% transform the goals into the agent's coordinate frame
% amgf = zeros(size(mgf));
aSgf = zeros(size(Sgf));

amgf = tfmtoa(mgf);
for i = 1:ntest
    aSgf(1:2,1:2,i) = rotm2d(rot_angle) * Sgf(1:2,1:2,i) * rotm2d(rot_angle)';
end

m_pi = zeros(ntest, Epi);
S_pi = zeros(ntest, Epi, Epi);
Cga_pi = zeros(ntest, Dpi, Epi);

m_f = zeros(ntest, Ef);
S_f = zeros(ntest, Ef, Ef);
Cax_f = zeros(ntest, Df, Ef);

% sSpi = diag([diag(chol_S_goal).^2; 0*ones(Dpi-length(chol_S_goal), 1)]);

for i = 1:ntest    
    s_g = amgf(1:2, i); S_g = aSgf(1:2, 1:2, i);
    sxi = [cos(gaze_angle); sin(gaze_angle)]; Sxi = zeros(2);    
    if gazestd == 0;
        sxi = zeros(0, ntest); Sxi = zeros(0);
    end

    % evaluate the GPs
    [m_pi(i,:), S_pi(i,:,:), Cga_pi(i,:,:), m_f(i,:), S_f(i,:,:), Cax_f(i,:,:)] = ...
        fwdInfGP(Xpi, xpi, ypi, Xf, xf, yf, s_g, S_g, sxi, Sxi);
end

%% ----------------------- compute errors ---------------------------------

sd = []; sdg = [];

fprintf('Forward inference angle RMSE: \n');
rmse = sqrt(mean((soa(1,:,1) - fwd_m_pi(:,1)').^2)) *180/pi;
rmseg = sqrt(mean((aasog - fwd_m_pi(:,1)').^2)) *180/pi;
disp([rmse, sd; rmseg, sdg]');

fprintf('Forward inference gaze vector RMSE: \n');
rmse = sqrt(mean((atan2(sox(2,:,1), sox(1,:,1)) - atan2(fwd_m_f(:,2)',fwd_m_f(:,1)')).^2)) *180/pi;
rmseg = sqrt(mean((asog - atan2(fwd_m_f(:,2)',fwd_m_f(:,1)')).^2)) *180/pi;
disp([rmse, sd; rmseg, sdg]');

fprintf('Reverse inference angle RMSE: \n');
rmse = sqrt(mean((soa(1,:,1) - maf(1,:)).^2)) *180/pi;
rmseg = sqrt(mean((aasog - maf(1,:)).^2)) *180/pi;
disp([rmse, sd; rmseg, sdg]');

fprintf('Reverse inference distance RMSE: \n');
rmse = (sqrt(mean((sod(1,:,1) - maf(2,:)).^2)));
rmseg = (sqrt(mean((dsog - maf(2,:)).^2)));
disp([rmse, sd; rmseg, sdg]');

fprintf('Reverse inference goal position RMSE: \n');
rmse = (sqrt(mean((tsgp(:,:,1) - amgf(1:2,:)).^2,2)));
rmseg = (sqrt(mean((tsog - amgf(1:2,:)).^2,2)));
disp([rmse', sd; rmseg', sdg]');

fprintf('Reverse inference goal position angle RMSE (mentor coords): \n');
rmse = sqrt(mean((atan2(sgp(2,:,1), sgp(1,:,1)) - atan2(mgf(2,:), mgf(1,:))).^2)) * 180/pi;
rmseg = sqrt(mean((asog - atan2(mgf(2,:), mgf(1,:))).^2)) * 180/pi;
disp([rmse, sd; rmseg, sdg]');

fprintf('Reverse inference goal position angle RMSE (agent coords): \n');
rmse = sqrt(mean((atan2(tsgp(2,:,1), tsgp(1,:,1)) - atan2(amgf(2,:), amgf(1,:))).^2)) * 180/pi;
rmseg = sqrt(mean((atsog - atan2(amgf(2,:), amgf(1,:))).^2)) * 180/pi;
disp([rmse, sd; rmseg, sdg]');

fprintf('Reverse+Forward inference gaze vector RMSE: \n');
rmse = sqrt(mean((atan2(tsox(2,:,1), tsox(1,:,1)) - atan2(m_f(:,2)',m_f(:,1)')).^2)) *180/pi;
rmseg = sqrt(mean((atsog - atan2(m_f(:,2)',m_f(:,1)')).^2)) *180/pi;
disp([rmse, sd; rmseg, sdg]');

%% ------------------------ plot results ----------------------------------

fprintf('plotting for obs %d\n', no);

ap = agentpos(1:2)';
mp = mentorpos(1:2)';

for fig = 1:9
try
    figure(fig); clf; hold on;
%     boxsize = [2 2];
%     rectangle('Position', [(ap - boxsize/2) boxsize], 'facecolor', 'c', 'Curvature', [1 1]);
%     rectangle('Position', [(mp - boxsize/2) boxsize], 'facecolor', 'g', 'Curvature', [1 1]);    
    if fig == 1
        title('angles (forward)');
        plot([cos(asog); cos(fwd_m_pi(:,1)'+asx0)], ...
            [sin(asog); sin(fwd_m_pi(:,1)'+asx0)], 'y');
        plot([cos(mean(soa, 3)+asx0); cos(fwd_m_pi(:,1)'+asx0)], ...
            [sin(mean(soa, 3)+asx0); sin(fwd_m_pi(:,1)'+asx0)], 'c:');
        plot(cos(xsa), sin(xsa), 'g. ');
        plot(cos(mean(soa, 3)+asx0), sin(mean(soa, 3)+asx0), 'b+ ');
        plot(cos(asog), sin(asog), 'ko ');
        plot(cos(fwd_m_pi(:,1)'+asx0), sin(fwd_m_pi(:,1)'+asx0), 'rx ');

    elseif fig == 2
        title('gaze vectors (forward)');
        plot([nsog(1, :); fwd_m_f(:, 1)'], ...
            [nsog(2, :); fwd_m_f(:, 2)'], 'y');
        plot([mean(sox(1,:,:),3); fwd_m_f(:, 1)'], ...
            [mean(sox(2,:,:),3); fwd_m_f(:, 2)'], 'c:');
        plot(sx(1, :), sx(2, :), 'g. ');
        plot(mean(sox(1,:,:),3), mean(sox(2,:,:),3), 'b+ ');
        plot(nsog(1,:), nsog(2,:), 'ko ');
        plot(fwd_m_f(:, 1), fwd_m_f(:, 2), 'rx ');

    elseif fig == 3
        title('angles (reverse)');
        plot([cos(asog); cos(maf(1,:)+asx0)], ...
            [sin(asog); sin(maf(1,:)+asx0)], 'y');
        plot([cos(mean(soa, 3)+asx0); cos(maf(1,:)+asx0)], ...
            [sin(mean(soa, 3)+asx0); sin(maf(1,:)+asx0)], 'c:');
        plot(cos(xsa), sin(xsa), 'g. ');
        plot(cos(mean(soa, 3)+asx0), sin(mean(soa, 3)+asx0), 'b+ ');
        plot(cos(asog), sin(asog), 'ko ');
        plot(cos(maf(1,:)+asx0), sin(maf(1,:)+asx0), 'rx ');
        
    elseif fig == 4
        title('goal positions (reverse)');
        plot([sog(1, :); mgf(1, :)], ...
            [sog(2, :); mgf(2, :)], 'y');
        plot([sgp(1,:,1); mgf(1, :)], ...
            [sgp(2,:,1); mgf(2, :)], 'c:');
        plot(sg(1, :), sg(2, :), 'g. ');
        plot(sgp(1, :, 1), sgp(2, :, 1), 'b+ ');
        plot(sog(1, :), sog(2, :), 'ko ');
        plot(mgf(1, :), mgf(2, :), 'rx ');

    elseif fig == 5
        title('goal positions (reverse) - mentor coords');
        plot([tsog(1, :); amgf(1, :)], ...
            [tsog(2, :); amgf(2, :)], 'y');
        plot([tsgp(1,:,1); amgf(1, :)], ...
            [tsgp(2,:,1); amgf(2, :)], 'c:');
        plot(tsg(1, :), tsg(2, :), 'g. ');
        plot(tsgp(1, :, 1), tsgp(2, :, 1), 'b+ ');
        plot(tsog(1, :), tsog(2, :), 'ko ');
        plot(amgf(1, :), amgf(2, :), 'rx ');
        
    elseif fig == 6
        title('gaze vectors (reverse + forward)');
        plot([ntsog(1,:); m_f(:, 1)'], ...
            [ntsog(2,:); m_f(:, 2)'], 'y');
        plot([mean(tsox(1,:,:),3); m_f(:, 1)'], ...
            [mean(tsox(2,:,:),3); m_f(:, 2)'], 'c:');
        plot(tsx(1, :), tsx(2, :), 'g. ');
        plot(mean(tsox(1,:,:),3), mean(tsox(2,:,:),3), 'b+ ');
        plot(ntsog(1, :), ntsog(2, :), 'ko ');
        plot(m_f(:, 1), m_f(:, 2), 'rx ');
    
    elseif fig == 7
%         title('distance from optimal gaze vector (forward)');
        angfwdmf = atan2(fwd_m_f(:,2)',fwd_m_f(:,1)');
        angdist = (asg - angfwdmf) * 180/pi;
        edges7 = floor(min(angdist)):ceil(max(angdist));
        vals7 = histc(angdist, edges7) / ntest;
%         bar(edges7, vals7);
        bar(-edges7, vals7); % flip to make the graph prettier
        
        xlabel('gaze error (degrees)');
        ylabel('p(error)');
%         histaxis = [-inf inf 0 0.2];
        histaxis = [-50 15 0 0.2];
        axis(histaxis);
%         set(gca,'XTick',[-45:15:45])
%         set(gca,'YTick',0:0.2:1)

        
        
    elseif fig == 8
%         title('distance from optimal gaze vector (reverse)');
        angmgf = atan2(amgf(2,:),amgf(1,:));
        angdist = (atsg - angmgf) * 180/pi;
        edges8 = floor(min(angdist)):ceil(max(angdist));
        vals8 = histc(angdist, edges8) / ntest;
        bar(edges8, vals8);
        
        xlabel('gaze error (degrees)');
        ylabel('p(error)');
        axis(histaxis);
        
    elseif fig == 9
%         title('distance from optimal gaze vector (gaze following: reverse + forward)');
        angmf = atan2(m_f(:,2)',m_f(:,1)');
        angdist = (atsg - angmf) * 180/pi;
        edges9 = floor(min(angdist)):ceil(max(angdist));
        vals9 = histc(angdist, edges9) / ntest;
        bar(edges9, vals9);
        
        xlabel('gaze error (degrees)');
        ylabel('p(error)');
        axis(histaxis);
        
    end
%     axis square; 
    hold off;
catch e
    % do nothing on error, just keep drawing
    fprintf('exception occurred when drawing figure %d: \n\t%s\n', fig, e.message);
end
end
drawnow;

end % nobs