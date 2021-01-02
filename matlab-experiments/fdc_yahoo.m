function viewsLbls = fdc_yahoo(DATAMAT)
    nrViews = 20;
    k = 40;
    shp = size(DATAMAT);

    shuffledCols = randperm(shp(2));
    colBatchSz = floor(shp(2)/nrViews);

    viewsLbls = zeros(nrViews, shp(1));
    j = 1;
    for i = 1:colBatchSz+1:shp(2)
        chosenCols_i = shuffledCols(i:min(i+colBatchSz, shp(2)));
        x = DATAMAT(:, chosenCols_i);
        colSums = sum(x, 1);%sum of values foreach column
        x = x(:,colSums > 0); % deletes empty columns
        nrValidCols = size(x,2);
        assert(nrValidCols > 0)
        cl_i = clkmeans(x, k, 'simcosi');
        viewsLbls(j,:) = cl_i;
        j = j + 1;
    end
    