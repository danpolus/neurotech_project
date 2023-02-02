%
%  inputs:
%       EEG
%       plotFlg
%
%  outputs:
%       AvalancheResults
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function AvalancheResults = avalanche_analysis(EEG, plotFlg)

%prepare global params for FindAvalanches
global param
param.Fs = EEG.srate;
%Event Size - range of avalanche sizes
param.ES.min = 1;
param.ES.max = (floor(EEG.nbchan*2/10))*10; %used to be 100 instead of 10
param.ES.edges = unique(ceil(logspace(0,log10(param.ES.max),25))); % log spaced bins
%times for raster plot zoom
param.t1 = 40; % sec
param.t2 = 50; % sec

project_params = doc_nft_params();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% zscore
EEG.data = normalize(EEG.data,2,'zscore');

% % plot
% if plotFlg
%     EEG_plot = pop_eegthresh(EEG, 1, 1:EEG.nbchan, -project_params.features.avlch_std_TH, project_params.features.avlch_std_TH, EEG.xmin, EEG.xmax, 0, 0);
%     pop_eegplot(EEG_plot, 1, 0, 0, [], 'srate',EEG_plot.srate, 'winlength',10, 'spacing', 5, 'eloc_file', []);
% end
    
%find avalanches
tau_vec = 1/EEG.srate:1/EEG.srate:project_params.features.tau_max_sec; 
AvalancheResults = FindAvalanches(EEG.data, EEG.times, tau_vec, 'maxmin', project_params.features.avlch_std_TH, 1, plotFlg);
tau = num2cell(tau_vec);
[AvalancheResults(:).tau] = deal(tau{:});
