%
clear all; close all;

project_params = doc_nft_params();

addpath(genpath([project_params.code_fp '\nft']));
fp = [project_params.data_fp '\mat\selected_mat\'];
% fp = [project_params.data_fp '\..\Epilepsy\'];

mkdir([fp 'temp\']);

plot_flg = false;

% rng(17); %set random seed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[files, in_fp] = uigetfile([fp '*.mat'], 'Select data files', 'MultiSelect','on');
if ~iscell(files) %in case only 1 file selected
    files = {files};
end
nFiles = length(files);

Results = repmat(struct('Name',''),nFiles,1);

% poolobj = gcp; %parpool
datetime(now,'ConvertFrom','datenum')
for iFile = 1:nFiles   %parfor
%     rng("shuffle"); %parfor
    filedata = load([in_fp files{iFile}]);
    EEGsect = filedata.EEGsect;

    Results(iFile).Name = EEGsect.setname;
    Results(iFile).chanlocs = EEGsect.chanlocs;
    Results(iFile).project_params = project_params;
    Results(iFile).FeaturesExperiment = extract_features(EEGsect, project_params.features.applyLaplacian_flg, plot_flg);
    try
        [~,fitted_elec_inx] = min([EEGsect.chanlocs.radius]); %Cz electrode
        [Results(iFile).NFTparams, Results(iFile).Spectra, Results(iFile).Chisq] = fit_nft(EEGsect, project_params, fitted_elec_inx, plot_flg);  
    catch
        warning([files{iFile} ':  fit_nft error']);
        continue;
    end
end
for iFile = 1:nFiles %(this part is not suitable to parfor)
%     rng("shuffle"); %parfor
    if isempty(Results(iFile).NFTparams)
        continue;
    end
    [EEGsim, Results(iFile).SimSpatialSpectra,~,Results(iFile).isSimSuccess] = simulate_nft(Results(iFile).NFTparams, Results(iFile).Spectra, project_params, iFile, plot_flg);
    if ~Results(iFile).isSimSuccess
        s=rng; rng(randi(100));
        [EEGsim, Results(iFile).SimSpatialSpectra,~,Results(iFile).isSimSuccess] = simulate_nft(Results(iFile).NFTparams, Results(iFile).Spectra, project_params, iFile, plot_flg);
        rng(s);
    end
    Results(iFile).Spectra = [];
    if ~Results(iFile).isSimSuccess
        continue;
    end

%     %use parfor%%%%%%%%%%%%%
%     save([fp 'temp\EEGsim' num2str(iFile) '.mat'], 'EEGsim');
% end
% parfor iFile = 1:nFiles 
%     if isempty(Results(iFile).NFTparams) || ~Results(iFile).isSimSuccess
%         continue;
%     end
%     filedata = load([fp 'temp\EEGsim' num2str(iFile) '.mat']);
%     EEGsim = filedata.EEGsim;
%     %use parfor%%%%%%%%%%%%%

    %post-processing
    EEGsim = pop_eegfiltnew(EEGsim, 'locutoff',project_params.pipelineParams.notchHz-2,'hicutoff',project_params.pipelineParams.notchHz+2,'revfilt',1); %notch filter
    EEGsim = pop_eegfiltnew(EEGsim, project_params.pipelineParams.passBandHz{1}, project_params.pipelineParams.passBandHz{2}); %high-pass (band pass) filter
    EEGsim = pop_reref(EEGsim,[]); %AVERAGE REFERENCE

    Results(iFile).FeaturesSimulation = extract_features(EEGsim, false, plot_flg);
end
datetime(now,'ConvertFrom','datenum')
% delete(poolobj);

save([fp 'Results_' datestr(now,'yyyymmdd_HHMM') '.mat'],'Results');
