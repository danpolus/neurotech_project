clear
close all

% Loading the data
eeglab_path = 'C:\Users\User\Desktop\eeglab\eeglab2022.1';
data_dir = 'C:\Users\User\Desktop\Rambam collaboration\009';


path_for_feature = 'C:\Users\User\Desktop\FeaturesToolbox-master';
addpath(path_for_feature)


ext = '*.set';
addpath(eeglab_path);
eeglab nogui;

dircrory = fullfile(data_dir, ext);
data_files = dir(dircrory);

for file=1:length(data_files)
    
    file_name = data_files(file).name; 

    EEG = pop_loadset('filename', file_name, 'filepath',data_dir);

    (Feat.Criticality.compute_avalanche(EEG))
        [a, b, c]=compute_avalanche(EEG);

end

%Feat.Criticality.compute_avalanche(EEG)
