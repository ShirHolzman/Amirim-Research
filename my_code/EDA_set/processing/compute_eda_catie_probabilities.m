% compute_eda_catie_probabilities.m
%
% Manager/Wrapper Script — EDA CATIE Choice Probability Calculation
%
% Iterates over each participant in the EDA SET and computes the CATIE model's
% predicted probability for the exact choice the participant made at each trial.
% All CATIE model logic is delegated entirely to the original source files:
%   COMPETITION_CATIE_schedule_choice_probability_hetro.m  (entry point)
%   COMPETITION_CATIE_schedule_choice_probability.m        (called internally)
%   CATIE_implementation_helpers/getExploreProb.m
%   CATIE_implementation_helpers/base2dec.m
%
% The data-mapping pattern (biased_reward → rewards_1, unbiased_reward → rewards_2,
% is_biased_choice → is_choice_1) is taken directly from the original study's
% COMPETITION_empirical_decisions_probabilities.m (lines 57–58).
%
% Input:  EDA_set/cleaned_eda_data.csv          (492 subjects × 100 trials)
% Output: catie_results/eda_with_catie_probabilities.csv  (same rows + 1 new column)
%         catie_results/README.md

%% ── 1. Path setup ────────────────────────────────────────────────────────────

% mfilename and getActiveDocument both return a temp path during section-mode
% execution (Ctrl+Enter). Instead, locate the project root by scanning upward
% from pwd — pwd is always a real path regardless of how MATLAB runs the code.
% The project root is the folder that contains 'Data_resources/'.
search = pwd;
project_root = '';
for k = 1:8
    if exist(fullfile(search, 'Data_resources'), 'dir')
        project_root = search;
        break;
    end
    parent = fileparts(search);
    if strcmp(parent, search), break; end  % reached filesystem root
    search = parent;
end
if isempty(project_root)
    error('Cannot find project root. cd into the Amirim Research folder (or any subfolder) and re-run.');
end

script_dir = fullfile(project_root, 'my_code', 'EDA_set', 'processing');
repo_root  = fullfile(project_root, 'Data_resources', 'competition_analysis-main');

addpath(fullfile(repo_root, 'CATIE'));
addpath(fullfile(repo_root, 'CATIE', 'CATIE_implementation_helpers'));

%% ── 2. Load EDA dataset ──────────────────────────────────────────────────────

% cleaned_eda_data.csv lives one level up in EDA_set/
eda_path = fullfile(script_dir, '..', 'cleaned_eda_data.csv');
fprintf('Reading EDA dataset: %s\n', eda_path);

data = readtable(eda_path, 'TextType', 'string');

n_rows     = height(data);
subjects   = unique(data.subject_file);
n_subjects = numel(subjects);
fprintf('Loaded %d rows | %d subjects | %d trials per subject.\n', ...
    n_rows, n_subjects, n_rows / n_subjects);

% Pre-allocate the output column (filled trial-by-trial in the loop below)
data.catie_choice_probability = NaN(n_rows, 1);

%% ── 3. Iterate over subjects ─────────────────────────────────────────────────
%
% For each subject:
%   a) Retrieve their rows and sort into chronological (trial_number) order.
%   b) Extract the three CATIE inputs using the same column mapping as the
%      original study's COMPETITION_empirical_decisions_probabilities.m:
%        rewards_1   = biased_reward   (reward for the "biased" alternative, each trial)
%        rewards_2   = unbiased_reward (reward for the "unbiased" alternative, each trial)
%        is_choice_1 = is_biased_choice (TRUE when subject chose the biased option)
%   c) Delegate to COMPETITION_CATIE_schedule_choice_probability_hetro — no
%      model logic lives in this script.
%   d) Write the returned 100×1 probability vector back into the main table.

fprintf('Processing %d subjects...\n', n_subjects);

for s = 1:n_subjects
    subject_id = subjects(s);
    mask       = data.subject_file == subject_id;
    orig_idx   = find(mask);                         % row indices in main table

    % Sort into chronological order (trial_number is 0-indexed: 0 … 99)
    [~, sort_perm] = sort(data.trial_number(orig_idx));
    sorted_idx = orig_idx(sort_perm);                % chronologically ordered indices

    % ── Map EDA columns → CATIE function arguments ────────────────────────
    rewards_1   = data.biased_reward(sorted_idx);                       % 100×1 double
    rewards_2   = data.unbiased_reward(sorted_idx);                     % 100×1 double
    % EDA CSV stores 'TRUE'/'FALSE'; original code used strcmp(...,'true').
    % strcmpi handles both cases safely.
    is_choice_1 = strcmpi(data.is_biased_choice(sorted_idx), 'true');  % 100×1 logical

    % ── Delegate entirely to the original CATIE function ──────────────────
    p_decisions = COMPETITION_CATIE_schedule_choice_probability_hetro( ...
        rewards_1, rewards_2, is_choice_1);          % returns 100×1 column vector

    % Write results back to the original (unsorted) row positions
    [~, inv_perm] = sort(sort_perm);
    data.catie_choice_probability(orig_idx) = p_decisions(inv_perm);

    if mod(s, 50) == 0 || s == n_subjects
        fprintf('  [%d / %d] subjects processed\n', s, n_subjects);
    end
end

%% ── 4. Verify output ─────────────────────────────────────────────────────────

n_nan     = sum(isnan(data.catie_choice_probability));
n_oor     = sum(data.catie_choice_probability < 0 | data.catie_choice_probability > 1);
mean_p    = mean(data.catie_choice_probability, 'omitnan');
mean_logp = mean(log(data.catie_choice_probability), 'omitnan');

% Trial 0 (first trial of each subject): CATIE always assigns p = 0.5
trial0_rows  = data.trial_number == 0;
trial0_probs = data.catie_choice_probability(trial0_rows);
n_wrong_t0   = sum(abs(trial0_probs - 0.5) > 1e-10);

fprintf('\n── Verification ───────────────────────────────────────────\n');
fprintf('  Rows with NaN probability  : %d  (expected 0)\n',   n_nan);
fprintf('  Rows outside [0, 1]        : %d  (expected 0)\n',   n_oor);
fprintf('  E[p]                       : %.4f  (paper ~0.619 for CATIE schedules)\n', mean_p);
fprintf('  E[log(p)]                  : %.4f  (paper ~-0.678 for CATIE schedules)\n', mean_logp);
fprintf('  Trial-0 rows with p != 0.5 : %d  (expected 0)\n',   n_wrong_t0);

if n_nan > 0 || n_oor > 0
    warning('compute_eda_catie_probabilities:unexpectedValues', ...
        'Output contains unexpected values. Inspect before using.');
end

%% ── 5. Save output CSV ───────────────────────────────────────────────────────

% Output lives alongside this script in the processing/ folder
result_dir = script_dir;
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

output_csv = fullfile(result_dir, 'eda_with_catie_probabilities.csv');
writetable(data, output_csv);

fprintf('\nOutput saved:\n  %s\n', output_csv);
fprintf('  Columns: %d  |  Rows: %d\n', width(data), height(data));
