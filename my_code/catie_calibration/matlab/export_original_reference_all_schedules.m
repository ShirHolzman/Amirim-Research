% export_original_reference_all_schedules.m
%
% Runs the ORIGINAL (unmodified, published) CATIE likelihood -- real MATLAB,
% no Python involved -- on every subject across all 12 reward schedules, and
% writes their per-trial P(choice actually made) to a CSV. This is the ground
% truth that golden_test.py's verify_against_live_matlab() diffs the Python
% port against.
%
% Two functions are called per subject, both unmodified originals from
%   Data_resources/competition_analysis-main/CATIE/
%     COMPETITION_CATIE_schedule_choice_probability_hetro.m   (k in {0,1,2} mixture)
%     COMPETITION_CATIE_schedule_choice_probability.m         (single k, k=0,1,2 each)
% Both already return P(the choice actually made) -- see their own final
% lines (`if is_choice_1(trial): p else 1-p`) -- so no conversion is applied
% here. (An earlier script in this folder, run_bug_comparison_all_schedules.m,
% documents a bug where that conversion was mistakenly applied a second time;
% see its header comment. Not repeated here.)
%
% Coverage: all 12 schedules, the same "kept" population sanitize_splits.py
% produces -- drop the curators' "..._INVALID_BIAS.csv" files, require exactly
% 100 trials -- 3,332 subjects.
%
% Output: results/original_reference_all_schedules.csv, long format, one row
% per subject x trial (333,200 rows). NOT tracked in git (see .gitignore) --
% regenerate by re-running this script.
%
% Run (from anywhere, non-interactively):
%   matlab -batch "run('my_code/catie_calibration/matlab/export_original_reference_all_schedules.m')"
%
% Takes a few minutes for ~3,300 subjects x 4 model calls each (1 hetero + 3
% per-k) -- comparable to run_bug_comparison_all_schedules.m.

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

this_dir     = fullfile(project_root, 'my_code', 'catie_calibration', 'matlab');
original_dir = fullfile(project_root, 'Data_resources', 'competition_analysis-main', 'CATIE');
results_dir  = fullfile(this_dir, 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

addpath(fullfile(original_dir, 'CATIE_implementation_helpers'));  % base2dec, getExploreProb
addpath(original_dir);   % COMPETITION_CATIE_schedule_choice_probability(_hetro), UNMODIFIED

fprintf('project root   : %s\n', project_root);
fprintf('original CATIE : %s\n\n', original_dir);

%% ── 2. Schedule directories (identical list to run_bug_comparison_all_schedules.m) ──

schedules = {
    'schedule_0',  fullfile(project_root, 'my_code', 'Schedule0_set', 'schedule_0')
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
K_VALUES = [0, 1, 2];

%% ── 3. Process every schedule, every subject ─────────────────────────────

out_rows = {};   % accumulate row structs, written once at the end
n_excluded_total = 0;
n_kept_total = 0;

for si = 1:size(schedules, 1)
    sched_label = schedules{si, 1};
    sched_dir   = schedules{si, 2};

    if ~exist(sched_dir, 'dir')
        fprintf('[%s] directory not found: %s -- skipping\n', sched_label, sched_dir);
        continue;
    end

    files = dir(fullfile(sched_dir, '*.csv'));
    fprintf('[%s] %d files in %s\n', sched_label, numel(files), sched_dir);

    n_kept = 0;
    n_excl = 0;

    for fi = 1:numel(files)
        fname = files(fi).name;
        if contains(lower(fname), 'invalid_bias')
            n_excl = n_excl + 1;
            continue;
        end
        fpath = fullfile(sched_dir, fname);

        try
            d = readtable(fpath, 'TextType', 'string');
        catch err
            fprintf('  !! read error %s: %s\n', fname, err.message);
            n_excl = n_excl + 1;
            continue;
        end

        req = {'trial_number','is_choice_alternative_1','reward_alternative_1','reward_alternative_2'};
        if ~all(ismember(req, d.Properties.VariableNames))
            n_excl = n_excl + 1;
            continue;
        end

        if height(d) ~= N_TRIALS || ~isequal(sort(d.trial_number), (0:N_TRIALS-1)')
            n_excl = n_excl + 1;
            continue;
        end

        [~, ord] = sort(d.trial_number);
        d = d(ord, :);

        rewards_1   = double(d.reward_alternative_1);
        rewards_2   = double(d.reward_alternative_2);
        is_choice_1 = strcmpi(strtrim(d.is_choice_alternative_1), 'true');

        % ── run the ORIGINAL, unmodified MATLAB -- both the mixture and each
        % single-k model -- exactly as published, bug included ──────────────
        pc_hetero = COMPETITION_CATIE_schedule_choice_probability_hetro(rewards_1, rewards_2, is_choice_1);
        pc_k = zeros(N_TRIALS, numel(K_VALUES));
        for ki = 1:numel(K_VALUES)
            pc_k(:, ki) = COMPETITION_CATIE_schedule_choice_probability(rewards_1, rewards_2, is_choice_1, K_VALUES(ki));
        end

        n_kept = n_kept + 1;
        subject_id = sprintf('%s/%s', sched_label, fname);
        for trial = 1:N_TRIALS
            row = struct();
            row.schedule      = sched_label;
            row.subject_id    = subject_id;
            row.subject_file  = fname;
            row.trial_number  = trial - 1;   % 0-indexed, matches the raw CSVs
            row.pc_hetero     = pc_hetero(trial);
            row.pc_k0         = pc_k(trial, 1);
            row.pc_k1         = pc_k(trial, 2);
            row.pc_k2         = pc_k(trial, 3);
            out_rows{end+1} = row; %#ok<AGROW>
        end
    end

    fprintf('  kept %d, excluded %d\n', n_kept, n_excl);
    n_kept_total = n_kept_total + n_kept;
    n_excluded_total = n_excluded_total + n_excl;
end

fprintf('\ntotal subjects kept: %d   total excluded: %d   total trial-rows: %d\n', ...
    n_kept_total, n_excluded_total, numel(out_rows));

%% ── 4. Write results ─────────────────────────────────────────────────────

T = struct2table([out_rows{:}]);
dest = fullfile(results_dir, 'original_reference_all_schedules.csv');
writetable(T, dest);
fprintf('\nwrote %s (%d rows)\n', dest, height(T));
fprintf('done.\n');
