function views = create_yahoo_views(DATAMAT)
    nrViews = 20;
    shp = size(DATAMAT);
    k = 40;

    shuffledCols = randperm(shp(2));
    colBatchSz = floor(shp(2)/nrViews);
    j = 1;
    dataViews = {};
    labels = {};
    for i = 1:colBatchSz+1:shp(2)
        chosenCols_i = shuffledCols(i:min(i+colBatchSz, shp(2)));
        x = DATAMAT(:, chosenCols_i);
        colSums = sum(x, 1);%sum of values foreach column
        x = x(:,colSums > 0); % deletes empty columns
        nrValidCols = size(x,2);
        assert(nrValidCols > 0);
        cl_i = clkmeans(x, k, 'simcosi');
        dataViews {j} = x;
        labels{j} = cl_i;
        j = j + 1;
    end
    views = {};
    views{1} = dataViews;
    views{2} = labels;
end