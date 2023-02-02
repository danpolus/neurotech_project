% computes valious EEG features
%
% inputs:
%       EEG - EEGLAB struct
%       applyLaplacian_flg - if true extract features from CSD
%       plot_flg - flag
% outputs:
%       Features - struct
%
function Features = extract_features(EEG, applyLaplacian_flg, plot_flg)

project_params = doc_nft_params();

if applyLaplacian_flg
    [G,H] = GetGH(ExtractMontage({EEG.chanlocs.labels}',[EEG.chanlocs.sph_theta]',[EEG.chanlocs.sph_phi]',{EEG.chanlocs.labels}'));
    EEG.data = CSD(single(EEG.data),G,H);
    %Henderson & Robinson 2006, https://doi.org/10.1016/j.ijpsycho.2015.04.023
end

if plot_flg  
    channel_map_topoplot(EEG, [], false);
end

Features = [];

Features.bad_channels = find(ismember({EEG.chanlocs.labels},EEG.bad_channels));

[P, f] = compute_psd('welch', EEG.data, EEG.srate, project_params.psd.window_sec, project_params.psd.overlap_percent,...
    [project_params.features.bands{2,1}(1)  project_params.features.bands{2,end}(2)], false);
Pnorm = P./sum(P,2);
Plog = log10(Pnorm);

Features.Spectra.P = P;
Features.Spectra.f = f;

%band power
Features.BandPowerdB = project_params.features.bands(1,:);
for iBand = 1:size(project_params.features.bands,2)
    Features.BandPowerdB{2,iBand} = ...
        sum(Pnorm(:,f>=project_params.features.bands{2,iBand}(1) & f<project_params.features.bands{2,iBand}(2)),2);
end

%spectral entropy
Features.SpectralEntropy = -sum(log2(Pnorm).*Pnorm, 2);

%spectral slope
logF = log10(f(f>=project_params.features.slopeFitFreqRange(1) & f<=project_params.features.slopeFitFreqRange(2)));
for iElec = 1:size(Plog,1)
    p(iElec,:) = polyfit(logF, Plog(iElec,f>=project_params.features.slopeFitFreqRange(1) & f<=project_params.features.slopeFitFreqRange(2)), 1);
end
Features.SpectralSlope = p(:,1); 

% "Robust EEG-based cross-site and cross-protocol classification of states of consciousness" Engemann et al
%permutation entropy 
% pe_window = project_params.psd.window_sec*EEG.srate; %  sliding window
for iElec = 1:size(EEG.data,1)
    Features.permEntropy(iElec) = pec(EEG.data(iElec,:),project_params.features.pe_order,project_params.features.pe_delay);
%     Features.permEntropy(iElec) = mean(PE(EEG.data(iElec,:),project_params.features.pe_delay,project_params.features.pe_order,pe_window));
end

%0-1 chaos
EEG_filtered = pop_eegfiltnew(EEG, project_params.features.chaos01passBandHz(1), project_params.features.chaos01passBandHz(2));
EEG_filtered = pop_resample(EEG_filtered, project_params.features.chaos01resampleFsHz);
for iElec = 1:size(EEG_filtered.data,1)
%     Features.chaos01(iElec) = z1test(EEG_filtered.data(iElec,:));
    Features.chaos01(iElec) = z1test_mex_mex(double(EEG_filtered.data(iElec,:)));
end

% % for LZ and avalanches that account for electrodes combination
% EEG.data(Features.bad_channels,:) = []; %remove bad channels
% maxNbadChan = floor(2*project_params.pipelineParams.max_badchannel_percent*EEG.nbchan);
% randInx = randperm(size(EEG.data,1));
% EEG.data = EEG.data(randInx(1:EEG.nbchan-maxNbadChan),:); %remove channels randomly down to a maximum number of bad channels

%LZ complexity
Features.LZcomplexity = mean(eeg2lzc(EEG.data),2);
% Features.LZcomplexity = mean(eeg2lzc_rows(EEG.data),2);

%neuronal avalanches
Features.AvalancheResults = avalanche_analysis(EEG, plot_flg);
Features.AvalancheResults = rmfield(Features.AvalancheResults,{'av_raster','av_label','av_size_vec','sigma_vec','av_dur_vec'});   
Features.AvalancheResults(1).alphaMsCr = polyfit([Features.AvalancheResults.tau],[Features.AvalancheResults.alpha],1);
Features.AvalancheResults(1).sigmaMsCr = polyfit([Features.AvalancheResults.tau],[Features.AvalancheResults.sigma],1);
Features.AvalancheResults(1).alpsigMsCr = polyfit([Features.AvalancheResults.sigma],[Features.AvalancheResults.alpha],1);

% functional_connectivity();
% weighted symbolic mutual information (wSMI), Engemann et al
