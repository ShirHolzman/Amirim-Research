% export_bma_mixing_inputs.m
%
% Exports the INPUT and OUTPUT of the original MATLAB's Bayesian-model-averaging
% (k-mixture) step, so the mixing arithmetic alone can be diffed against
% catie_core.mix_agents() on IDENTICAL inputs.
%
% Why this is a different test from golden_test.py check 1:
%   golden_test compares the END of the whole pipeline (state recursion +
%   per-trial probability + BMA mixing) against MATLAB. If the state recursion
%   and the mixing had compensating errors, that check would still pass.
%   This script isolates the mixing step: it hands Python the exact per-agent
%   probability matrix MATLAB itself built, so the ONLY thing being compared
%   is the BMA arithmetic in hetro.m:20-25 vs mix_agents().
%
% Calls, unmodified:
%   COMPETITION_CATIE_schedule_choice_probability.m       (per k, the BMA input)
%   COMPETITION_CATIE_schedule_choice_probability_hetro.m (the BMA output)
% and reproduces hetro.m's own K = 0:2 loop (lines 6-17) only to capture the
% intermediate matrix that hetro.m computes internally but does not return.
%
% NOTE ON PROBABILITY SPACE -- this is the substantive thing under test.
% COMPETITION_CATIE_schedule_choice_probability.m returns P(CHOICE ACTUALLY
% MADE), not P(alt 1) -- see its lines 136-139. So hetro.m does its weighted
% mix in choice-space. catie_core.mix_agents() instead mixes in P(alt 1) space
% and converts afterwards. Those agree only because the BMA weights sum to 1
% (mixing then flipping == flipping then mixing, for a convex combination).
% This export is what lets that equivalence be checked numerically rather than
% argued on paper.
%
% Output: results/bma_mixing_reference.csv
%   schedule, subject, trial, is_choice_1, p_k0, p_k1, p_k2, p_hetro
%   (p_k* and p_hetro are all in P(choice made) space, as MATLAB produces them)
%
% Run:
%   matlab -batch "run('my_code/catie_calibration/matlab/export_bma_mixing_inputs.m')"

%% -- 1. Path setup (same search as report_original_elogp.m) ----------------

search = pwd;
project_root = '';
for i = 1:8
    if exist(fullfile(search, 'Data_resources'), 'dir')
        project_root = search;
        break;
    end
    parent = fileparts(search);
    if strcmp(parent, search), break; end
    search = parent;
end
if isempty(project_root)
    error('Cannot find project root. cd into the Amirim Research folder (or any subfolder) and re-run.');
end

original_dir = fullfile(project_root, 'Data_resources', 'competition_analysis-main', 'CATIE');
addpath(fullfile(original_dir, 'CATIE_implementation_helpers'));
addpath(original_dir);

out_dir = fullfile(project_root, 'my_code', 'catie_calibration', 'matlab', 'results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
out_csv = fullfile(out_dir, 'bma_mixing_reference.csv');

fprintf('project root   : %s\n', project_root);
fprintf('original CATIE : %s\n', original_dir);
fprintf('output         : %s\n\n', out_csv);

%% -- 2. Schedules (same list/order as report_original_elogp.m) -------------

schedules = {
    fullfile(project_root, 'my_code', 'Schedule0_set', 'schedule_0')
    fullfile(project_root, 'my_code', 'Test_set', 'schedule_1')
    fullfile(project_root, 'my_code', 'Training_set', 'schedule_2')
    fullfile(project_root, 'my_code', 'Training_set', 'schedule_3')
    fullfile(project_root, 'my_code', 'EDA_set', 'schedule_4')
    fullfile(project_root, 'my_code', 'EDA_set', 'schedule_5')
    fullfile(project_root, 'my_code', 'Training_set', 'schedule_6')
    fullfile(project_root, 'my_code', 'EDA_set', 'schedule_7')
    fullfile(project_root, 'my_code', 'Test_set', 'schedule_8')
    fullfile(project_root, 'my_code', 'Training_set', 'schedule_9')
    fullfile(project_root, 'my_code', 'Test_set', 'schedule_10')
    fullfile(project_root, 'my_code', 'Training_set', 'schedule_11')
    };

N_TRIALS       = 100;
K              = 0:2;      % hetro.m:6 -- reproduced here only to capture the input matrix
MAX_PER_SCHED  = 50;       % evenly spaced sample; the arithmetic is subject-independent

%% -- 3. Score sampled subjects, capturing BOTH the BMA input and output ----

rows_sched  = strings(0,1);
rows_subj   = strings(0,1);
rows_trial  = [];
rows_choice = [];
rows_pk     = [];   % n x 3
rows_hetro  = [];

n_subj = 0;
for si = 1:numel(schedules)
    sched_dir = schedules{si};
    [~, sched_name] = fileparts(sched_dir);
    if ~exist(sched_dir, 'dir')
        fprintf('%s not found -- skipping\n', sched_dir);
        continue;
    end
    files = dir(fullfile(sched_dir, '*.csv'));

    % valid files first, then evenly spaced subsample
    keep = true(numel(files),1);
    for fi = 1:numel(files)
        if contains(lower(files(fi).name), 'invalid_bias'), keep(fi) = false; end
    end
    files = files(keep);
    if numel(files) > MAX_PER_SCHED
        idx = round(linspace(1, numel(files), MAX_PER_SCHED));
        files = files(unique(idx));
    end

    n_before = n_subj;
    for fi = 1:numel(files)
        fname = files(fi).name;
        d = readtable(fullfile(sched_dir, fname), 'TextType', 'string');
        req = {'trial_number','is_choice_alternative_1','reward_alternative_1','reward_alternative_2'};
        if ~all(ismember(req, d.Properties.VariableNames)), continue; end
        if height(d) ~= N_TRIALS || ~isequal(sort(d.trial_number), (0:N_TRIALS-1)'), continue; end

        [~, ord] = sort(d.trial_number);
        d = d(ord, :);
        rewards_1   = double(d.reward_alternative_1);
        rewards_2   = double(d.reward_alternative_2);
        is_choice_1 = strcmpi(strtrim(d.is_choice_alternative_1), 'true');

        % --- the BMA INPUT: hetro.m:14-17, verbatim loop over the ORIGINAL fn
        agents = zeros(numel(K), N_TRIALS);
        for ai = 1:numel(K)
            agents(ai,:) = COMPETITION_CATIE_schedule_choice_probability( ...
                               rewards_1, rewards_2, is_choice_1, K(ai));
        end

        % --- the BMA OUTPUT: the ORIGINAL, unmodified hetro function
        p_hetro = COMPETITION_CATIE_schedule_choice_probability_hetro( ...
                      rewards_1, rewards_2, is_choice_1);

        rows_sched  = [rows_sched;  repmat(string(sched_name), N_TRIALS, 1)];   %#ok<AGROW>
        rows_subj   = [rows_subj;   repmat(string(fname),      N_TRIALS, 1)];   %#ok<AGROW>
        rows_trial  = [rows_trial;  (0:N_TRIALS-1)'];                           %#ok<AGROW>
        rows_choice = [rows_choice; double(is_choice_1(:))];                    %#ok<AGROW>
        rows_pk     = [rows_pk;     agents'];                                   %#ok<AGROW>
        rows_hetro  = [rows_hetro;  p_hetro(:)];                                %#ok<AGROW>
        n_subj = n_subj + 1;
    end
    fprintf('[%-12s] %d subjects (running total %d)\n', sched_name, n_subj - n_before, n_subj);
end

%% -- 4. Write, at full double precision -----------------------------------
% Deliberately NOT writetable: it serialises doubles at ~15 significant digits,
% which would put a floor of ~1e-16 relative error on the comparison and could
% mask (or manufacture) a discrepancy at exactly the scale being tested.
% %.17g round-trips an IEEE double exactly.

fid = fopen(out_csv, 'w');
fprintf(fid, 'schedule,subject,trial,is_choice_1,p_k0,p_k1,p_k2,p_hetro\n');
for r = 1:numel(rows_trial)
    fprintf(fid, '%s,%s,%d,%d,%.17g,%.17g,%.17g,%.17g\n', ...
            rows_sched(r), rows_subj(r), rows_trial(r), rows_choice(r), ...
            rows_pk(r,1), rows_pk(r,2), rows_pk(r,3), rows_hetro(r));
end
fclose(fid);

fprintf('\nwrote %d subjects x %d trials = %d rows\n', n_subj, N_TRIALS, numel(rows_trial));
fprintf('%s\n', out_csv);
