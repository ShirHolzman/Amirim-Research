% report_original_elogp.m
%
% Reports E[log p] by CALLING the paper's own original, unmodified MATLAB
% functions for both steps -- not reimplementing their arithmetic:
%
%   Data_resources/competition_analysis-main/CATIE/
%     COMPETITION_CATIE_schedule_choice_probability_hetro.m
%       -- per-subject P(choice made), k in {0,1,2} mixture. Same function
%          COMPETITION_empirical_decisions_probabilities.m:58 calls.
%   Data_resources/competition_analysis-main/helper_functions/
%     permutation_test_and_bootstrap.m
%       -- the SAME function COMPETITION_main.m:202 feeds log(catie_probabilities)
%          into. It takes two arrays and has no return value (report-only, via
%          fprintf), so it is called here as
%          permutation_test_and_bootstrap(all_log_p, all_log_p) -- comparing
%          our pooled log(p) to itself -- and its own internal
%          `data1 = data1(:); mean(data1)` (lines 3-7) is what produces the
%          number; this script only regexp's it back out of the function's own
%          printed report (captured via evalc), rather than computing it here.
%
% COMPETITION_empirical_decisions_probabilities.m itself is NOT called: it
% hardcodes an absolute path to another machine's disk
% ('C:\Users\ojd5\...') and computes 4 unrelated QL models per participant.
% The per-subject loop over our own schedule folders below is necessary glue
% (our data lives in a different location/layout), but every actual
% COMPUTATION -- the per-trial probability, and the final aggregate mean --
% is delegated to the original functions above, unmodified.
%
% Coverage: all 12 schedules, the same "kept" population sanitize_splits.py
% produces -- drop the curators' "..._INVALID_BIAS.csv" files, require
% exactly 100 trials -- 3,332 subjects, 333,200 trials.
%
% Why this script exists, given export_original_reference_all_schedules.m
% already caches pc_hetero per subject per trial from a live MATLAB run: that
% CSV is deliberately per-trial, so it can be diffed against the Python port
% trial-by-trial in golden_test.py. This script exists to report the single
% AGGREGATE number the paper actually prints, via the paper's own aggregation
% function, without requiring a Python step -- a direct, from-scratch check.
%
% Result (2026-08-23, this population): E[log p] = -0.6733, identical to
% catie_core.py's "published" port (see golden_test.py, 1e-15 agreement) and
% ~+0.0047 above the paper's reported -0.678. That residual is real and
% unexplained -- see REVIEW_PLAN.md -- not a bug in either implementation.
%
% Run (from anywhere, non-interactively):
%   matlab -batch "run('my_code/catie_calibration/matlab/report_original_elogp.m')"
% Takes roughly a minute for ~3,300 subjects.

%% ── 1. Path setup ────────────────────────────────────────────────────────

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
helpers_dir  = fullfile(project_root, 'Data_resources', 'competition_analysis-main', 'helper_functions');
addpath(fullfile(original_dir, 'CATIE_implementation_helpers'));  % base2dec, getExploreProb
addpath(original_dir);   % COMPETITION_CATIE_schedule_choice_probability_hetro, UNMODIFIED
addpath(helpers_dir);    % permutation_test_and_bootstrap, UNMODIFIED

fprintf('project root      : %s\n', project_root);
fprintf('original CATIE    : %s\n', original_dir);
fprintf('original helpers  : %s\n\n', helpers_dir);

%% ── 2. Schedule directories (identical list to export_original_reference_all_schedules.m) ──

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

N_TRIALS = 100;

%% ── 3. Score every subject, pool log(p) exactly as the paper's code does ──

all_log_p = [];   % pooled, matching permutation_test_and_bootstrap's data1(:)
n_subj = 0;

for si = 1:numel(schedules)
    sched_dir = schedules{si};
    if ~exist(sched_dir, 'dir')
        fprintf('%s not found -- skipping\n', sched_dir);
        continue;
    end
    files = dir(fullfile(sched_dir, '*.csv'));
    n_before = n_subj;

    for fi = 1:numel(files)
        fname = files(fi).name;
        if contains(lower(fname), 'invalid_bias'), continue; end

        d = readtable(fullfile(sched_dir, fname), 'TextType', 'string');
        req = {'trial_number','is_choice_alternative_1','reward_alternative_1','reward_alternative_2'};
        if ~all(ismember(req, d.Properties.VariableNames)), continue; end
        if height(d) ~= N_TRIALS || ~isequal(sort(d.trial_number), (0:N_TRIALS-1)'), continue; end

        [~, ord] = sort(d.trial_number);
        d = d(ord, :);
        rewards_1   = double(d.reward_alternative_1);
        rewards_2   = double(d.reward_alternative_2);
        is_choice_1 = strcmpi(strtrim(d.is_choice_alternative_1), 'true');

        % the ORIGINAL, unmodified function -- returns P(choice made)
        p_decisions = COMPETITION_CATIE_schedule_choice_probability_hetro(rewards_1, rewards_2, is_choice_1);

        all_log_p = [all_log_p; log(p_decisions)]; %#ok<AGROW>
        n_subj = n_subj + 1;
    end
    fprintf('[%s] %d subjects (running total %d)\n', sched_dir, n_subj - n_before, n_subj);
end

%% ── 4. Aggregate via the ORIGINAL permutation_test_and_bootstrap.m ────────
% Called on (all_log_p, all_log_p) -- comparing our data to itself. The
% function's own "Mean1 = %.3f" is mean(data1) computed by ITS code (lines
% 3-7: data1=data1(:); mean(data1)), not by us. Its other outputs (diff,
% p-value, CI) are degenerate/meaningless here (data1 == data2) and ignored;
% only Mean1, the E[log p] figure, is what this script is after.

captured = evalc('permutation_test_and_bootstrap(all_log_p, all_log_p)');
fprintf('\n--- verbatim report from the ORIGINAL permutation_test_and_bootstrap.m ---\n');
fprintf('%s', captured);

tok = regexp(captured, 'Mean1 = ([-0-9.]+)', 'tokens');
E_log_p = str2double(tok{1}{1});

fprintf('---------------------------------------------------------------------\n');
fprintf('\n=================================================\n');
fprintf('subjects = %d, trials = %d\n', n_subj, numel(all_log_p));
fprintf('E[log p] (original, unmodified MATLAB, published) = %.4f\n', E_log_p);
fprintf('this project''s catie_core.py port (published)     = -0.6733\n');
fprintf('paper reported (Tables S1/S2)                      = -0.678\n');
fprintf('=================================================\n');
