addpath( genpath('./ClusterPack-V2.0'))

load('data/yahoo-all/yahoo-all.mat')
trLbls = importTrueLabels('data/yahoo-all/docs.int.labels')';

cls = fdc_yahoo(doc_term_matrix);

cl = clusterensemble(cls, 20); %'cspa', 'hgpa', 'mcla'

disp(['Consensus clustering has a mutual info ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(1,:))) ' for CSPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(2,:))) ' for HGPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(3,:))) ' for MCLA ']);

cspa = cl(1,:);
hgpa = cl(2,:);
mcla = cl(3,:);
save('yahoo_ensemble_results.mat', 'cspa','hgpa','mcla','trLbls');
% Consensus clustering has a mutual info 
% of 0.39298 for CSPA 
% of 0.18665 for HGPA 
% of 0.3133 for MCLA 

clear trLbls cls cl doc_term_matrix dataset;

%% Yahoo included in HDF5 format.
trLbls = double(h5read('data/yahoo_coo.h5', '/labels')');
%k = length(unique(trLbls)) * 2;
k = 40;
viewsLbls = zeros(20, length(trLbls));
for i = 0:19
    viewPath = sprintf('/views/v%d/coo-format/',i)
    hw_rows = h5read('data/yahoo_coo.h5', [viewPath 'rowindex']);
    hw_cols = h5read('data/yahoo_coo.h5', [viewPath 'colindex']);
    hw_data = h5read('data/yahoo_coo.h5', [viewPath 'data']);

    hwsp = sparse(hw_rows+1, hw_cols+1, double(hw_data));    
    x = full(hwsp);
    cl_i = clkmeans(x, k, 'simcosi');
    viewsLbls(i+1,:) = cl_i;
    clear hw_rows hw_cols hw_data hwsp x cl_i viewPath;
end
cl = clusterensemble(viewsLbls, 20);
disp(['Consensus clustering has a mutual info ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(1,:))) ' for CSPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(2,:))) ' for HGPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(3,:))) ' for MCLA ']);

cspa = cl(1,:);
hgpa = cl(2,:);
mcla = cl(3,:);
save('yahoo_ensemble_results.mat', 'cspa','hgpa','mcla', 'trLbls');

% Consensus clustering has a mutual info 
% of 0.41636 for CSPA 
% of 0.19392 for HGPA 
% of 0.33025 for MCLA 

clear i cl viewsLbls k trLbls viewNames cspa hgpa mcla;
%% Too slow!
trLbls = double(h5read('data/reuters_coo.h5', '/labels')');
k = length(unique(trLbls)) * 2;
viewNames = {'rt_english','rt_france','rt_german','rt_italian','rt_spanish'}
viewsLbls = zeros(length(viewNames), length(trLbls));
for i = 1:length(viewNames)
    viewPath = sprintf('/views/%s/coo-format/',viewNames{i})
    hw_rows = h5read('data/reuters_coo.h5', [viewPath 'rowindex']);
    hw_cols = h5read('data/reuters_coo.h5', [viewPath 'colindex']);
    hw_data = h5read('data/reuters_coo.h5', [viewPath 'data']);

    hwsp = sparse(hw_rows+1, hw_cols+1, hw_data);    
    x = full(hwsp);
    cl_i = clkmeans(x, k, 'simcosi');
    viewsLbls(i,:) = cl_i;
    clear hw_rows hw_cols hw_data hwsp x cl_i viewPath;
end
cl = clusterensemble(viewsLbls, 10);
disp(['Consensus clustering has a mutual info ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(1,:))) ' for CSPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(2,:))) ' for HGPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(3,:))) ' for MCLA ']);

cspa = cl(1,:);
hgpa = cl(2,:);
mcla = cl(3,:);
save('reuters_ensemble_results.mat', 'cspa','hgpa','mcla','trLbls');

% Consensus clustering has a mutual info 
% of XXX for CSPA 
% of XXX for HGPA 
% of XXX for MCLA 

clear i cl viewsLbls k trLbls viewNames cspa hgpa mcla;
%%
trLbls = double(h5read('data/bbc-seg4_coo.h5', '/labels')');
k = length(unique(trLbls)) * 2;
viewNames = {'bbc4_v0','bbc4_v1','bbc4_v2','bbc4_v3'}
viewsLbls = zeros(length(viewNames), length(trLbls));

for i = 1:length(viewNames)
    viewPath = sprintf('/views/%s/coo-format/',viewNames{i})
    hw_rows = h5read('data/bbc-seg4_coo.h5', [viewPath 'rowindex']);
    hw_cols = h5read('data/bbc-seg4_coo.h5', [viewPath 'colindex']);
    hw_data = h5read('data/bbc-seg4_coo.h5', [viewPath 'data']);

    hwsp = sparse(hw_rows+1, hw_cols+1, hw_data);    
    x = full(hwsp);
    cl_i = clkmeans(x, k, 'simcosi');
    viewsLbls(i+1,:) = cl_i;
    clear bbc4_rows bbc4_cols bbc4_data bbc4sp x cl_i viewPath;
end
cl = clusterensemble(viewsLbls, 10);
disp(['Consensus clustering has a mutual info ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(1,:))) ' for CSPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(2,:))) ' for HGPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(3,:))) ' for MCLA ']);

cspa = cl(1,:);
hgpa = cl(2,:);
mcla = cl(3,:);
save('bbc4_ensemble_results.mat', 'cspa','hgpa','mcla','trLbls');

% Consensus clustering has a mutual info 
% of 0.45577 for CSPA 
% of 0.35498 for HGPA 
% of 0.40884 for MCLA 

clear i cl viewsLbls k trLbls cspa hgpa mcla;
%%
trLbls = double(h5read('data/handwritten_coo.h5', '/labels')');
k = length(unique(trLbls)) * 2;
viewNames = {'hw_fac','hw_fou','hw_kar','hw_mor','hw_pix','hw_zer'}
viewsLbls = zeros(length(viewNames), length(trLbls));
for i = 1:length(viewNames)
    viewPath = sprintf('/views/%s/coo-format/',viewNames{i})
    hw_rows = h5read('data/handwritten_coo.h5', [viewPath 'rowindex']);
    hw_cols = h5read('data/handwritten_coo.h5', [viewPath 'colindex']);
    hw_data = h5read('data/handwritten_coo.h5', [viewPath 'data']);

    hwsp = sparse(hw_rows+1, hw_cols+1, hw_data);    
    x = full(hwsp);
    cl_i = clkmeans(x, k, 'simcosi');
    viewsLbls(i,:) = cl_i;
    clear hw_rows hw_cols hw_data hwsp x cl_i viewPath;
end
cl = clusterensemble(viewsLbls, 10);
disp(['Consensus clustering has a mutual info ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(1,:))) ' for CSPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(2,:))) ' for HGPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(3,:))) ' for MCLA ']);

cspa = cl(1,:);
hgpa = cl(2,:);
mcla = cl(3,:);
save('handwritten_ensemble_results.mat', 'cspa','hgpa','mcla','trLbls');
% Consensus clustering has a mutual info 
% of 0.76564 for CSPA 
% of 0.50381 for HGPA 
% of 0.68445 for MCLA 

clear i cl viewsLbls k trLbls viewNames cspa hgpa mcla;
%% 
trLbls = double(h5read('data/caltech-20_coo.h5', '/labels')');
k = length(unique(trLbls)) * 2;
viewNames = {'ct20_centrist','ct20_gabor','ct20_gist','ct20_hog','ct20_lbp','ct20_wm'}
viewsLbls = zeros(length(viewNames), length(trLbls));
for i = 1:length(viewNames)
    viewPath = sprintf('/views/%s/coo-format/',viewNames{i})
    hw_rows = h5read('data/caltech-20_coo.h5', [viewPath 'rowindex']);
    hw_cols = h5read('data/caltech-20_coo.h5', [viewPath 'colindex']);
    hw_data = h5read('data/caltech-20_coo.h5', [viewPath 'data']);

    hwsp = sparse(hw_rows+1, hw_cols+1, hw_data);    
    x = full(hwsp);
    cl_i = clkmeans(x, k, 'simcosi');
    viewsLbls(i,:) = cl_i;
    clear hw_rows hw_cols hw_data hwsp x cl_i viewPath;
end
cl = clusterensemble(viewsLbls, 10);
disp(['Consensus clustering has a mutual info ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(1,:))) ' for CSPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(2,:))) ' for HGPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(3,:))) ' for MCLA ']);

cspa = cl(1,:);
hgpa = cl(2,:);
mcla = cl(3,:);
save('caltech-20_ensemble_results.mat', 'cspa','hgpa','mcla','trLbls');

% Consensus clustering has a mutual info 
% of 0.44739 for CSPA 
% of 0.35931 for HGPA 
% of 0.4711 for MCLA 

clear i cl viewsLbls k trLbls viewNames cspa hgpa mcla;
%% 
trLbls = double(h5read('data/nusWide_coo.h5', '/labels')');
k = length(unique(trLbls)) * 2;
viewNames = {'nw_ch','nw_cm','nw_corr','nw_edh','nw_wt'}
viewsLbls = zeros(length(viewNames), length(trLbls));
for i = 1:length(viewNames)
    viewPath = sprintf('/views/%s/coo-format/',viewNames{i})
    hw_rows = h5read('data/nusWide_coo.h5', [viewPath 'rowindex']);
    hw_cols = h5read('data/nusWide_coo.h5', [viewPath 'colindex']);
    hw_data = h5read('data/nusWide_coo.h5', [viewPath 'data']);

    hwsp = sparse(hw_rows+1, hw_cols+1, hw_data);    
    x = full(hwsp);
    cl_i = clkmeans(x, k, 'simcosi');
    viewsLbls(i,:) = cl_i;
    clear hw_rows hw_cols hw_data hwsp x cl_i viewPath;
end
cl = clusterensemble(viewsLbls, 10);
disp(['Consensus clustering has a mutual info ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(1,:))) ' for CSPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(2,:))) ' for HGPA ']);
disp(['of ' num2str(evalmutual(trLbls ,cl(3,:))) ' for MCLA ']);

cspa = cl(1,:);
hgpa = cl(2,:);
mcla = cl(3,:);
save('nusWide_ensemble_results.mat', 'cspa','hgpa','mcla','trLbls');

% Consensus clustering has a mutual info 
% of 0.096239 for CSPA 
% of 0.001147 for HGPA 
% of 0.11223 for MCLA 

clear i cl viewsLbls k trLbls viewNames cspa hgpa mcla;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% EXPERIMENTS WITH NegMM ("Ensemble clustering based on evidence extracted...", 2019) %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath( genpath('./clustering-ensemble-zhongcaiming'))

trLbls = double(h5read('data/yahoo_coo.h5', '/labels'));

N = size(trLbls, 1);
M = 20;
PI = zeros(N, M);
for i = 0:19
    viewLabelPath = sprintf('/views/v%d/labels',i)
    lbls_i = double(h5read('data/yahoo_coo.h5', viewLabelPath));
    PI(:, i+1) = lbls_i;
end
negMMLabels = NegMM(PI, trLbls, 'Yahoo');
save('yahoo_negMM_results.mat','negMMLabels','trLbls');

clear i viewsLbls trLbls viewNames PI N M viewLabelPath lbls_i negMMLabels; 
%%
clear
trLbls = double(h5read('data/bbc-seg2_coo.h5', '/labels'));

viewNames = {'bbc2_v0','bbc2_v1'}

N = size(trLbls, 1);
M = length(viewNames);
PI = zeros(N, M);
for i = 1:length(viewNames)
    viewLabelPath = sprintf('/views/%s/labels',viewNames{i})
    lbls_i = double(h5read('data/bbc-seg2_coo.h5', viewLabelPath));
    PI(:, i) = lbls_i;
end

negMMLabels = NegMM(PI, trLbls, 'BBC-2');
save('bbc2_negMM_results.mat','negMMLabels','trLbls');

clear i viewsLbls trLbls viewNames PI N M viewLabelPath lbls_i negMMLabels;
%%
trLbls = double(h5read('data/bbc-seg4_coo.h5', '/labels'));

viewNames = {'bbc4_v0','bbc4_v1','bbc4_v2','bbc4_v3'}

N = size(trLbls, 1);
M = length(viewNames);
PI = zeros(N, M);
for i = 1:length(viewNames)
    viewLabelPath = sprintf('/views/%s/labels',viewNames{i})
    lbls_i = double(h5read('data/bbc-seg4_coo.h5', viewLabelPath));
    PI(:, i) = lbls_i;
end

negMMLabels = NegMM(PI, trLbls, 'BBC-4');
save('bbc4_negMM_results.mat','negMMLabels','trLbls');

clear i viewsLbls trLbls viewNames PI N M viewLabelPath lbls_i negMMLabels; 
%%
trLbls = double(h5read('data/handwritten_coo.h5', '/labels'));

viewNames = {'hw_fac','hw_fou','hw_kar','hw_mor','hw_pix','hw_zer'}

N = size(trLbls, 1);
M = length(viewNames);
PI = zeros(N, M);
for i = 1:length(viewNames)
    viewLabelPath = sprintf('/views/%s/labels',viewNames{i})
    lbls_i = double(h5read('data/handwritten_coo.h5', viewLabelPath));
    PI(:, i) = lbls_i;
end

negMMLabels = NegMM(PI, trLbls, 'Handwritten');
save('handwritten_negMM_results.mat','negMMLabels','trLbls');

clear i viewsLbls trLbls viewNames PI N M viewLabelPath lbls_i negMMLabels; 

%% 
trLbls = double(h5read('data/caltech-20_coo.h5', '/labels'));

viewNames = {'ct20_centrist','ct20_gabor','ct20_gist','ct20_hog','ct20_lbp','ct20_wm'}

N = size(trLbls, 1);
M = length(viewNames);
PI = zeros(N, M);
for i = 1:length(viewNames)
    viewLabelPath = sprintf('/views/%s/labels',viewNames{i})
    lbls_i = double(h5read('data/caltech-20_coo.h5', viewLabelPath));
    PI(:, i) = lbls_i;
end

negMMLabels = NegMM(PI, trLbls, 'Caltech-20');
save('caltech-20_negMM_results.mat','negMMLabels','trLbls');

clear i viewsLbls trLbls viewNames PI N M viewLabelPath lbls_i negMMLabels; 
%% Too much memory, couldn't finish (pc with ~20GB free ram)
trLbls = double(h5read('data/nusWide_coo.h5', '/labels'));

viewNames = {'nw_ch','nw_cm','nw_corr','nw_edh','nw_wt'}

N = size(trLbls, 1);
M = length(viewNames);
PI = zeros(N, M);
for i = 1:length(viewNames)
    viewLabelPath = sprintf('/views/%s/labels',viewNames{i})
    lbls_i = double(h5read('data/nusWide_coo.h5', viewLabelPath));
    PI(:, i) = lbls_i;
end

negMMLabels = NegMM(PI, trLbls, 'NusWide');
save('nusWide_negMM_results.mat','negMMLabels','trLbls');

clear i viewsLbls trLbls viewNames PI N M viewLabelPath lbls_i negMMLabels;
%% Too slow!
trLbls = double(h5read('data/reuters_coo.h5', '/labels'));

viewNames = {'rt_english','rt_france','rt_german','rt_italian','rt_spanish'}

N = size(trLbls, 1);
M = length(viewNames);
PI = zeros(N, M);
for i = 1:length(viewNames)
    viewLabelPath = sprintf('/views/%s/labels',viewNames{i})
    lbls_i = double(h5read('data/reuters_coo.h5', viewLabelPath));
    PI(:, i) = lbls_i;
end

negMMLabels = NegMM(PI, trLbls, 'Reuters');
save('reuters_negMM_results.mat','negMMLabels','trLbls');

clear i viewsLbls trLbls viewNames PI N M viewLabelPath lbls_i negMMLabels;