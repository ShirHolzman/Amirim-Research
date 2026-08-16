% export_state_tensors.m
%
% Exports MATLAB's own per-trial internal state variables -- the exact
% quantities my_code/catie_calibration/catie_core.py's state_tensors() is
% meant to reproduce -- for every subject across all 12 reward schedules and
% k in {0,1,2}, using CATIE_FIXED/COMPETITION_CATIE_schedule_choice_probability_INSTRUMENTED.m.
%
% This is a true element-by-element unit-level cross-check of state_tensors(),
% independent of every previous check in this project (golden_test.py,
% run_bug_comparison_all_schedules.m), which compared only the FINAL choice
% probability. It exports H, b, c_prev, s_prev, sbar_prev, g directly from
% MATLAB's own loop variables.
%
% Output: results/state_tensors_matlab.csv (long format: one row per
% subject x k x trial, ~1M rows). NOT tracked in git (large, regenerable) --
% see the .gitignore entry added alongside this script.
%
% Run:
%   matlab -batch "run('my_code/catie_calibration/matlab/export_state_tensors.m')"
%
% Then verify with:
%   python my_code/catie_calibration/matlab/verify_state_tensors.py

%% ── 1. Path setup (same pattern as run_bug_comparison_all_schedules.m) ──

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

this_dir     = fullfile(project_root, 'my_code', 'catie_calibration', 'matlab');
original_dir = fullfile(project_root, 'Data_resources', 'competition_analysis-main', 'CATIE');
fixed_dir    = fullfile(this_dir, 'CATIE_FIXED');
results_dir  = fullfile(this_dir, 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

addpath(fullfile(original_dir, 'CATIE_implementation_helpers'));  % base2dec, getExploreProb
addpath(fixed_dir);  % ..._INSTRUMENTED

fprintf('project root : %s\n', project_root);

%% ── 2. Schedule directories ──────────────────────────────────────────────

schedules = {
    'schedule_0',  fullfile(project_root, 'my_code', 'schedule_0')
    'schedule_1',  fullfile(project_root, 'my_code', 'Test_set', 'schedule_1')
    'schedule_2',  fullfile(project_root, 'my_code', 'Training_set', 'schedule_2')
    'schedule_3',  fullfile(project_root, 'my_code', 'Training_set', 'schedule_3')
    'schedule_4',  fullfile(project_root, 'my_code', 'EDA_set', 'schedule_4')
    'schedule_5',  fullfile(project_root, 'my_code', 'EDA_set', 'schedule_5')
    'schedule_6',  fullfile(project_root, 'my_code', 'Training_set', 'schedule_6')
    'schedule_7',  fullfile(project_root, 'my_code', 'EDA_set', 'schedule_7')
    'schedule_8',  fullfile(project_root, 'my_code', 'Test_set', 'schedule_8')
    'schedule_9',  fullfile(project_root, 'my_code', 'Training_set', 'schedule_9')
    'schedule_10', fullfile(project_root, 'my_code', 'Test_set', 'schedule_10')
    'schedule_11', fullfile(project_root, 'my_code', 'Training_set', 'schedule_11')
    };

N_TRIALS = 100;
MIN_CHOICES_PER_SIDE = 5;
KS = [0, 1, 2];

%% ── 3. Preallocate (upper bound: every raw file kept, x 100 trials x 3 k) ─

max_files = 0;
for si = 1:size(schedules, 1)
    if exist(schedules{si, 2}, 'dir')
        max_files = max_files + numel(dir(fullfile(schedules{si, 2}, '*.csv')));
    end
end
max_rows = max_files * N_TRIALS * numel(KS);
fprintf('preallocating for up to %d rows (%d files x %d trials x %d k-values)\n', ...
    max_rows, max_files, N_TRIALS, numel(KS));

sched_col    = strings(max_rows, 1);
subj_col     = strings(max_rows, 1);
k_col        = zeros(max_rows, 1);
trial_col    = zeros(max_rows, 1);
H_col        = zeros(max_rows, 1);
b_col        = zeros(max_rows, 1);
cprev_col    = zeros(max_rows, 1);
sprev_col    = zeros(max_rows, 1);
sbarprev_col = zeros(max_rows, 1);
g_col        = zeros(max_rows, 1);

cursor = 0;
n_subjects_done = 0;
trial_range = (1:N_TRIALS)';

%% ── 4. Process every schedule ────────────────────────────────────────────

for si = 1:size(schedules, 1)
    sched_label = schedules{si, 1};
    sched_dir   = schedules{si, 2};
    if ~exist(sched_dir, 'dir')
        fprintf('[%s] directory not found -- skipping\n', sched_label);
        continue;
    end

    files = dir(fullfile(sched_dir, '*.csv'));
    fprintf('[%s] %d files\n', sched_label, numel(files));
    n_kept = 0;

    for fi = 1:numel(files)
        fname = files(fi).name;
        if contains(lower(fname), 'invalid_bias'), continue; end
        fpath = fullfile(sched_dir, fname);

        try
            d = readtable(fpath, 'TextType', 'string');
        catch
            continue;
        end

        req = {'trial_number','is_biased_choice','side_choice','biased_reward','unbiased_reward'};
        if ~all(ismember(req, d.Properties.VariableNames)), continue; end

        [side_counts, ~] = groupcounts(d.side_choice);
        if numel(side_counts) < 2 || min(side_counts) < MIN_CHOICES_PER_SIDE, continue; end

        if height(d) ~= N_TRIALS || ~isequal(sort(d.trial_number), (0:N_TRIALS-1)')
            continue;
        end

        [~, ord] = sort(d.trial_number);
        d = d(ord, :);

        rewards_1   = double(d.biased_reward);
        rewards_2   = double(d.unbiased_reward);
        is_choice_1 = strcmpi(d.is_biased_choice, 'true');

        subject_id = sprintf('%s/%s', sched_label, fname);

        for kk = KS
            [~, H_out, b_out, c_prev_out, s_prev_out, sbar_prev_out, g_out] = ...
                COMPETITION_CATIE_schedule_choice_probability_INSTRUMENTED( ...
                    rewards_1, rewards_2, is_choice_1, kk);

            idx = (cursor+1):(cursor+N_TRIALS);
            sched_col(idx)    = sched_label;
            subj_col(idx)     = subject_id;
            k_col(idx)        = kk;
            trial_col(idx)    = trial_range;
            H_col(idx)        = H_out(:);
            b_col(idx)        = b_out(:);
            cprev_col(idx)    = c_prev_out(:);
            sprev_col(idx)    = s_prev_out(:);
            sbarprev_col(idx) = sbar_prev_out(:);
            g_col(idx)        = g_out(:);
            cursor = cursor + N_TRIALS;
        end

        n_kept = n_kept + 1;
        n_subjects_done = n_subjects_done + 1;
    end
    fprintf('  kept %d subjects\n', n_kept);
end

fprintf('\ntotal subjects: %d   total rows: %d\n', n_subjects_done, cursor);

%% ── 5. Trim and write ────────────────────────────────────────────────────

idx = 1:cursor;
T = table(sched_col(idx), subj_col(idx), k_col(idx), trial_col(idx), ...
    H_col(idx), b_col(idx), cprev_col(idx), sprev_col(idx), sbarprev_col(idx), g_col(idx), ...
    'VariableNames', {'schedule','subject_id','k','trial','H','b','c_prev','s_prev','sbar_prev','g'});

out_path = fullfile(results_dir, 'state_tensors_matlab.csv');
writetable(T, out_path);
fprintf('\nwrote %s\n  %d rows (%d subjects x %d k-values)\n', out_path, height(T), n_subjects_done, numel(KS));
fprintf('\ndone.\n');
