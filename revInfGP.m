function [maf, Saf, mgf, Sgf] = revInfGP(Xpi, xpi, ypi, Xf, xf, yf, s_g, S_g, sxi, Sxi, xf_obs, nsamples)

    Dpi = size(xpi, 2); Epi = size(ypi, 2);

    mg_filt = zeros(Dpi, nsamples);
    Sg_filt = zeros(Dpi, Dpi, nsamples);
    
    % compute predictive distributions p(a), p(x_f)
    [ma, Sa, Cga, mxf, Sxf, Cax, mpi, Spi] = ...
        fwdInfGP(Xpi, xpi, ypi, Xf, xf, yf, s_g, S_g, sxi, Sxi);
    
    % "filter" step for a: combine prediction and measurement
    Lf = chol(Sxf)'; Bf = Lf\(Cax');
    tmaf = Cax*(Sxf\(xf_obs-mxf));
    maf = ma + tmaf(1:Epi);
    tSaf = Bf'*Bf;
    Saf = Sa - tSaf(1:Epi,1:Epi);
    
    % sampled "filter" step for g: combine prediction and measurement
    % (samples drawn from the filter step for a)
    a_smpl = zeros(Epi, nsamples);
    for j = 1:nsamples
        % sample a from N(ma_filt, Sa_filt)
        sa = maf + (randn(1,Epi)*chol(Saf))';
        a_smpl(:,j) = sa;

        % apply "filter" step for this sample
        Lpi = chol(Sa)'; Bpi = Lpi\(Cga');
        mg_filt(:,j) = mpi + Cga*(Sa\(sa-ma));
        Sg_filt(:,:,j) = Spi - Bpi'*Bpi;
    end
    
    psa = mvnormpdf(a_smpl, maf, [], Saf);
    psa = psa ./ sum(psa);
    tg = bsxfun(@times, mg_filt, psa);
    mgf = sum(tg, 2);
    Sgf = sum(bsxfun(@times, Sg_filt(:, :, :), reshape(psa, 1, 1, [])), 3);
end