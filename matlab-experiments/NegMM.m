function bestSolution =  NegMM(Views, C_Label, datasetNm)
    clc;
    close all;
    addpath("clustering-ensemble-zhongcaiming")
    addpath("clustering-ensemble-zhongcaiming/Ncut")
    addpath("clustering-ensemble-zhongcaiming/Datasets")
   
    K = length(unique(C_Label)); % Number of clusters
    CM = GenCM(Views);    

    disp(['Data set: ',  strtrim(datasetNm)]);
    disp('================================');

    bestSolutionIndexValue = -1;
    bestSolution = [];
    for delta = 0.01: 0.01: 0.5
        CM1 = CM;
        CM1(CM1<=delta)=0;
        [C,~] = NcutClustering(CM1, K); 

        CA_value = CA(C, C_Label);
        MM_value = minimax(C, CM, 3);
        if (MM_value < bestSolutionIndexValue || bestSolutionIndexValue == -1)
            bestSolutionIndexValue = MM_value;
            bestSolution = C;
        end

        disp(['CA value: ', num2str(CA_value), ', MM value: ', num2str(MM_value) ]);
    end     
    %Draw(X, bestSolution);
    disp(['Best CA value: ', num2str(CA(bestSolution, C_Label)), ', MM value: ', num2str(bestSolutionIndexValue ) ]);
end

function NegMM_test
    clc;
    close all;
    addpath("clustering-ensemble-zhongcaiming")
    addpath("clustering-ensemble-zhongcaiming/Ncut")
    addpath("clustering-ensemble-zhongcaiming/Datasets")
    
    X = load(['Datasets/', strtrim('spiral'), '.txt']);
    C_Label = load(['Datasets/', strtrim('spiral'), '_label.txt']);

    K = length(unique(C_Label)); % Number of clusters
    M = 1000; % Number of base partitions: normally 500 is enough, but for some datasets with complex structures, 1000 will be more stable.
    It = 4; % Number of iterations of K-means      

    % Produce base partitions
    PI = BasePartitionByKmeans(X, M, It, 'Fixed'); 
    CM = GenCM(PI);    

    disp(['Data set: ',  strtrim('spiral')]);
    disp('================================');

    bestSolutionIndexValue = -1;
    bestSolution = [];
    for delta = 0.01: 0.01: 0.5
        CM1 = CM;
        CM1(CM1<=delta)=0;
        [C,~] = NcutClustering(CM1, K); 

        CA_value = CA(C, C_Label);
        MM_value = minimax(C, CM, 3);
        if (MM_value < bestSolutionIndexValue || bestSolutionIndexValue == -1)
            bestSolutionIndexValue = MM_value;
            bestSolution = C;
        end

        disp(['CA value: ', num2str(CA_value), ', MM value: ', num2str(MM_value) ]);
    end     
    Draw(X, bestSolution);
    disp(['Best CA value: ', num2str(CA(bestSolution, C_Label)), ', MM value: ', num2str(bestSolutionIndexValue ) ]);
end

%% Generate base partitions by K-means
% X: data set, one row is one instance
% M: the number of base partitions
% It: the number of iterations for K-means
% Ktype: the type to generate base partitions, 'Fixed' ---sqrt(N)
% PI: base partitions,one column is a partition
% ClusterNum: the number of K in each partition
function [PI] = BasePartitionByKmeans(X, M, It, Ktype)
    N = size(X, 1);
    PI = zeros(N, M);
    
    for i = 1: M
        if strcmp(Ktype,'Fixed')
            K = ceil(sqrt(N));
        else
            K = randsample(2:ceil(sqrt(N)),1);
        end

        opts = statset('MaxIter', It);        
        C = kmeans(X, K, 'emptyaction', 'drop', 'Options', opts);
        
        while length(unique(C)) ~= K
            C = kmeans(X, K, 'emptyaction', 'drop', 'Options', opts);
        end   
        PI(:, i) = C;      
    end
end

%% Generate Co-association Matrix
function CM = GenCM( PI)
    N = size(PI,1);
    CM = zeros(N);
    PiNo  = size(PI, 2);      
    
    for i = 1: PiNo
        C = PI(:,i);
        for j = 1: length(unique(PI(:,i)))
            IDX = find(C==j);
            if length(IDX) <=1 
                continue;
            end                  
            n = length(IDX);       
            Pairs = combntns(1:n,2);
            Ind = [IDX(Pairs(1:n*(n-1)/2)),IDX(Pairs(n*(n-1)/2+1:n*(n-1)))];
            CM((Ind(:,2) - 1)* N + Ind(:,1)) = CM((Ind(:,2) - 1)* N + Ind(:,1)) + 1;   
        end
    end    
    CM = (CM + CM') / size(PI,2) + eye(N);
end


%% Produce pathbased similarity
function CM = PathbasedSimi(W, k)
    N = size(W, 1);
    if k <= 10
        S = sumOfNeighbors(W, k);    
        W = W .* repmat(S', 1, N) .* repmat(S, N, 1);    
    end
    W = PathbasedDist(W.^(-1)) + eye(N);
    CM = W.^(-1);
end

function M = sumOfNeighbors(W, numOfNeighbours)
    M = sort(W, 'descend');
    M= sum(M(2:numOfNeighbours+1,:));
    M = M / max(M);
end

%% MM index
% C - Labels
% CM0 - Co-association matrix
% k - number of nearest neighbors
function s = minimax(C, CM0, k)
    CM = PathbasedSimi(CM0, k);
    s = 0;
    
    NumC = length(unique(C));
    for i = 1: NumC
        a = find(C == i);
        if length(a) <= k
            s = Inf;
            return;
        end
        s1 = max(max(CM(a, C ~= i)));
        
        CM0_a = CM0(a, a);
        CM_a = PathbasedSimi(CM0_a, k);
        
        try
            [C1, ~] = NcutClustering(CM_a, 2);
            flag = 1;
        catch e
            flag = 0;
        end
   
        if flag == 1
            s2 = min(min(CM_a(C1==1, C1==2)));
        else
            s2 = Inf;
        end
        s = s + s1 / s2;
    end
end