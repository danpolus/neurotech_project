clear; close all; clc
% ctrl + A - select all, ctrl + I - indent

user = 'Inbal';
% user = 'Oren';

% Set EEGLAB path and load data
if strcmp(user,'Inbal')
    eeglab_path = 'C:\Users\User\Desktop\eeglab\eeglab2022.1'; % Inbal
    data_dir = 'C:\Users\User\Desktop\Rambam collaboration';% Inbal
elseif strcmp(user,'Oren')
    eeglab_path = 'D:\Dropbox\eeglab2022.1'; % Oren
    data_dir = 'D:\Dropbox (BGU)\Research\BGU\Collaborations\Rambam collaboration\Rambam Data';% Oren
end

% set subject
subject = '009';

% The problematic part:
path_for_feature = 'C:\Users\User\Desktop\FeaturesToolbox-master\src';
addpath(path_for_feature)

ext = '*.set';
addpath(eeglab_path);
eeglab nogui;

dir_name= fullfile(data_dir,subject); % directory with files from a single subject
data_files = dir(fullfile(dir_name,ext));

%for file = 1:length(data_files)
file = 1;
file_name = data_files(file).name;

% load data
EEG = pop_loadset('filename', file_name, 'filepath',dir_name);

% sampling_rate = EEG.srate; % (Hz)
% t_vec = EEG.times; % times vector (msec)
% D = EEG.data; % data
%  n_comp =
bnd = [1 40]; % [low, high] frequency in Hz.

% Preprocess data
EEG_clean_data = Clean_data_Inbal_basic(EEG.data,bnd);


%     D_z_score = zscore(D')';
%
%     figure('Color','w');
%     subplot(2,1,1);plot(t_vec,D(1,:));xlabel('t (ms)');ylabel('V')
%     subplot(2,1,2);plot(t_vec,D_z_score(1,:));xlabel('t (ms)');ylabel('Z')


%     Perform avalanche analysis
%      (Feat.Criticality.compute_avalanche(EEG))
%          [a, b, c]=compute_avalanche(EEG);

%end

%Feat.Criticality.compute_avalanche(EEG)
