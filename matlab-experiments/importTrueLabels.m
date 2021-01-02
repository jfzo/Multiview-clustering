function docs = importTrueLabels(filename, dataLines)
%IMPORTFILE Import data from a text file
%  DOCS = IMPORTFILE(FILENAME) reads data from text file FILENAME for
%  the default selection.  Returns the numeric data.
%
%  DOCS = IMPORTFILE(FILE, DATALINES) reads data for the specified row
%  interval(s) of text file FILENAME. Specify DATALINES as a positive
%  scalar integer or a N-by-2 array of positive scalar integers for
%  dis-contiguous row intervals.
%
%  Example:
%  docs = importfile("/home/juan/Documentos/ensembles-matlab/data/yahoo-all/docs.int.labels", [1, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 11-Nov-2020 21:05:38

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [1, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 1);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = "VarName1";
opts.VariableTypes = "double";

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
docs = readtable(filename, opts);

%% Convert to output type
docs = table2array(docs);
end