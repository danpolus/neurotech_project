%doc nft analysis parameters
function project_params = doc_nft_params()

project_params.code_fp = '..';
project_params.data_fp = 'C:\My Files\Work\BGU\Datasets\Rambam collaboration\set';
project_params.minSectLenSec = 5*60; %minimal length for sufficient avalanche analysis
project_params.maxSectLenSec = 10*60; %maximal length 
project_params.head_radius = 0.5; %used to get rid of out-of-scalp channels

project_params.fs = 500; %%resampling and more
project_params.electrodes_fn = [project_params.data_fp '\chanlocs18.sfp'];


%%%PSD
project_params.psd.window_sec = 4; %Assadzadeh
project_params.psd.overlap_percent = 0.75; %Assadzadeh

%%% section
project_params.section.fftWindowSize = 2;
project_params.section.WindowForMeanSqeeze = 5;
project_params.section.ThresholdZscore = 2;
project_params.section.ThresholdChannel = 0.05; % 5 precent of the channels

%%%%eeglab pipelineParams
%filtering parameters:
pipelineParams.passBandHz = [{0.2}, {[]}];
pipelineParams.notchHz = 50;
pipelineParams.resampleFsHz = project_params.fs;
%bad channels and bad epochs parameters:
pipelineParams.badchan_time_percent = 1/3;
pipelineParams.badchan_window_step_length_sec = 5;
pipelineParams.chan_z_score_thresh = 7;
pipelineParams.epoch_max_std_thresh = 3;
pipelineParams.epoch_min_std_thresh = 0.1;
pipelineParams.bad_chan_in_epoch_percent = [];
%bad sections parameters:
pipelineParams.badsect_z_score_thresh = 5; %set to 7 to detect avalanches?
pipelineParams.badsect_window_length_sec = 1;
pipelineParams.badsect_reject_score_thresh = 1;
%clean_artifacts parameters:
pipelineParams.max_badchannel_percent = 0.1;
pipelineParams.minimal_interchannel_correlation = 0.6;
pipelineParams.channel_max_bad_time = pipelineParams.badchan_time_percent;
pipelineParams.asr_birst = 'off';
pipelineParams.window_criterion = max(1,pipelineParams.badsect_reject_score_thresh);
if isnumeric(pipelineParams.asr_birst)
    pipelineParams.badsect_z_score_thresh = pipelineParams.asr_birst+1;
end
%ica paremeters:
pipelineParams.ica_flg = true; %avoid ICA to detect avalanches?
pipelineParams.minimal_nonbrain_class_prob = 0.5;
pipelineParams.tweaks_ics2rjct_fun = [];
%various paremeters:
pipelineParams.elecInterpolateFn = [];%[project_params.code_fp '\Dynamic-States\eeglab\eeg_unitvector_62.txt';
pipelineParams.ref_electrodes = [];%re-reference to the average of T7,T8 to increase the contrast between 2 hemispheres
project_params.pipelineParams = pipelineParams;

%%%%Features
project_params.features.applyLaplacian_flg = false;
if isMEGdata
    project_params.features.applyLaplacian_flg = false;
end
project_params.features.bands = {'delta','theta','alpha','beta',   'gamma';...
                                  [1 4],  [4 8],  [8 12], [12 24], [24 40]};
project_params.features.slopeFitFreqRange = [12 40]; %beta+gamma
project_params.features.pe_order = 3; % permutation entropy order 4 of ordinal patterns (5-points ordinal patterns). typical 3-:-5. rule: 5*(order + 1)! < pe_window
project_params.features.pe_delay = 1; % permutation entropy delay 1 between points in ordinal patterns (successive points). typical 1-:-4
project_params.features.avlch_std_TH = 3.0;
project_params.features.tau_max_sec = 0.04; %avalanche analysis with different tau up to 40msec
project_params.features.chaos01passBandHz = [1 6];
project_params.features.chaos01resampleFsHz = 25;
project_params.features.chaos01alpha = 0.85;

%%%NFT fit
% project_params.nftfit.params2fit = {'Gee','Gei','Ges','Gse','Gsr','Gre','Grs','Alpha','Beta','t0', 'EMGa'};
project_params.nftfit.params2fit = {'Gee','Gei','Ges','Gse','Gsr','Gsn','Gre','Grs','Alpha','Beta','t0','EMGa',...
                                  'Gee_amp','Gei_amp','Gsn_amp','t0_amp','Alpha_amp','Beta_amp'}; 
% project_params.nftfit.params2fit = {'Gee','Gei','Ges','Gse','Gsr','Gsn','Gre','Grs','Alpha','Beta','t0','EMGa',...
%                                   'Gee_x_var','Gee_y_var','Gee_x_ph','Gee_y_ph'};%,...
%                                   'Gei_x_var','Gei_y_var','Gei_x_ph','Gei_y_ph',...
%                                   'Gsn_x_var','Gsn_y_var','Gsn_x_ph','Gsn_y_ph',...
%                                   'Alpha_x_var','Alpha_y_var','Alpha_x_ph','Alpha_y_ph',...
%                                   'Beta_x_var','Beta_y_var','Beta_x_ph','Beta_y_ph',...
%                                   't0_x_var','t0_y_var','t0_x_ph','t0_y_ph'};
% project_params.nftfit.params2fit = {'Ges'};
% project_params.nftfit.params2fit = {};
project_params.nftfit.spatial_fit_flg = true;
project_params.nftfit.psdMethod = 'fft';
project_params.nftfit.freqBandHz = [1 40]; %maybe narrow, to avoid poor fitting 
project_params.nftfit.weigths1f_flg = true;
project_params.nftfit.npoints = 1e5;
project_params.nftfit.chisqThrsh = 7; %for spatial fit warning

%%%%NFT simulation
project_params.nftsim.grid_edge = 28; % pi/4*(grid_edge/2)^2 >= number of experimental electrode inside head radius
project_params.nftsim.fs = project_params.fs*8; %1000Hz
project_params.nftsim.out_dt = 1/project_params.fs;
project_params.nftsim.params2explore = {'Gee',[0.0001 20]; 'Gei',[-20 -0.0001]; 'Ges',[0.0001 11]; 'Gse',[0.0001 17]; 'Gsr',[-9 -0.0001]; 'Gre',[0.0001 7]; 'Grs',[0.0001 8]; 'alpha',[10 120]; 'beta',[100 800]; 't0',[0.075 0.14]}; %param name and limits
% project_params.nftsim.params2explore = {'Gee',[16 19.5]; 'Gei',[-19.5 -16]; 'Ges',[0.0001 1]; 'Gse',[0.5 3.5]; 'Gsr',[-2.5 -0.5]; 'Gre',[0.5 3.5]; 'Grs',[1 8]; 't0',[0.075 0.14]}; %narrow range
project_params.nftsim.nGridSearchPnts = 10;
