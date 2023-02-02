%
clear all; close all;

project_params = doc_nft_params();

fp = project_params.data_fp;
in_folder = 'raw\';
out_folder = 'mat\';
% electrodes_fn = [fp 'electrodes\GSN-HydroCel-256.sfp'];

goodSectEvents = {'eyeo', 'eyeb', 'section', 'intra-section'};
badSectEvents = {'eyec'};

manual_statsect_flg = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
%EEG=pop_chanedit(EEG, 'lookup','C:\\My Files\\Work\\Software\\eeglab2020_0\\plugins\\dipfit\\standard_BESA\\standard-10-5-cap385.elp');
%

[files, in_fp] = uigetfile([[fp in_folder] '*.raw'], 'Select data files', 'MultiSelect','on');
if ~iscell(files) %in case only 1 file selected
    files = {files};
end

for iFile = 1:length(files)
    fn = files{iFile};

    % read raw data
    EEG = pop_fileio([in_fp fn], 'dataformat','auto');
    EEG.setname = fn(1:end-4); 
    EEG.filename = fn;
    EEG.filepath = [fp out_folder];
    EEG = eeg_checkset(EEG);

    % remove irrelevant channels
    if EEG.nbchan == 257
        EEG = pop_select(EEG, 'nochannel', 257);
    end
    EEG.chanlocs = readlocs(project_params.electrodes_fn);
    EEG = pop_select(EEG, 'nochannel', find([EEG.chanlocs.radius] > project_params.head_radius) ); %remove channels that out of head radius
    EEG.bad_channels = [];
    EEG = eeg_checkset(EEG);

    %preliminary preprocressing
    pipelineParams = project_params.pipelineParams;
    %notch filter
    EEG = pop_eegfiltnew(EEG, 'locutoff',pipelineParams.notchHz-2,'hicutoff',pipelineParams.notchHz+2,'revfilt',1);
    pipelineParams.notchHz = [];
    %high-pass (band pass) filter
    EEG = pop_eegfiltnew(EEG, pipelineParams.passBandHz{1}, pipelineParams.passBandHz{2});
    pipelineParams.passBandHz = [];
    %resample
    EEG = pop_resample(EEG, pipelineParams.resampleFsHz);
    pipelineParams.resampleFsHz = [];
    EEG = eeg_checkset(EEG);  
    
    if manual_statsect_flg
        %select stationary segment. default: full range
        pop_eegplot(EEG, 1, 0, 0, [], 'winlength',60, 'spacing', 50); pause;
        segmentRange_str = inputdlg('Enter time range in sec','',1);
        close all;
        segmentRange = str2num(segmentRange_str{1})*EEG.srate;
        EEG.event = [struct('duration',0, 'type','section', 'latency',segmentRange(1), 'urevent',0),...
                  struct('duration',0, 'type','section', 'latency',segmentRange(2), 'urevent',0)];
        events = squeeze(struct2cell(EEG.event));
    else
        %cut to sections according to events
        IdxBySec = find_stationaty_sections(EEG, project_params);
        [EEG, events] = add_to_event(EEG, IdxBySec, [goodSectEvents badSectEvents], project_params.maxSectLenSec);
    end
    
    nEvents = size(events,2);
    for iEvent = 1:nEvents
        if iEvent < nEvents
            endInx = events{3,iEvent+1}-1;
        else
            endInx = EEG.pnts;
        end
        if (endInx - events{3,iEvent} < project_params.minSectLenSec*EEG.srate) ||...
              contains(badSectEvents,events{2,iEvent})  
            continue;
        end
        EEGsect = pop_select(EEG, 'point',[events{3,iEvent} endInx]);
        eventTime = floor((events{3,iEvent}-1)/EEG.srate);
        EEGsect.setname = [EEG.setname ' ' num2str(eventTime) events{2,iEvent}]; 
        EEGsect.filename = [EEGsect.filename(1:end-4) '_' num2str(eventTime) events{2,iEvent} '.set'];
        EEGsect = eeg_checkset(EEGsect);
        EEGsect = eeglab_pipeline(EEGsect, pipelineParams, 0, 0);
        if EEGsect.xmax < project_params.minSectLenSec || numel(EEGsect.data) == 0 ||...
            length(EEGsect.bad_channels) > 2*pipelineParams.max_badchannel_percent*EEGsect.nbchan
            continue;
        end

        EEGsect = pop_saveset(EEGsect, 'filename',EEGsect.filename, 'filepath',EEGsect.filepath, 'savemode','onefile');
%         csvwrite([fullfile(EEGsect.filepath, EEGsect.filename(1:end-4)) '_' num2str(EEGsect.srate) 'Hz.csv'], EEGsect.data);
        save([fullfile(EEGsect.filepath, EEGsect.filename(1:end-4)) '_' num2str(EEGsect.srate) 'Hz.mat'], 'EEGsect');
    end
end
