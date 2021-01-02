%% Generate base partitions by K-means
% X: data set, one row is one instance
% M: the number of base partitions
% It: the number of iterations for K-means
% Ktype: the type to generate base partitions, 'Fixed' ---sqrt(N)
% PI: base partitions,one column is a partition
% ClusterNum: the number of K in each partition
function PI = BasePartitionByKmeans(X, M, It)
    N = size(X, 1);
    PI = zeros(N, M);                                     
    
    for i = 1: M
        K = ceil(sqrt(N));
        opts = statset('MaxIter', It);        
        C = kmeans(X, K, 'emptyaction', 'drop', 'Options', opts);
        while length(unique(C)) ~= K
            C = kmeans(X, K, 'emptyaction', 'drop', 'Options', opts);
        end   
        PI(:, i) = C;      
    end
end