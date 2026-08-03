% run_bug_comparison_all_schedules.m
%
% Runs the ORIGINAL (published) and FIXED CATIE likelihood -- both real
% MATLAB, no Python involved -- on every subject across all 12 reward
% schedules, and reports the E[p] / E[log p] impact of the heuristic-mode
% fix described in
%   my_code/catie_calibration/01_bug_correction/VERIFICATION_MEMO.md
%
% Original model:
%   Data_resources/competition_analysis-main/CATIE/
%     COMPETITION_CATIE_schedule_choice_probability_hetro.m
%     COMPETITION_CATIE_schedule_choice_probability.m
% Fixed model (this repo, only the heuristic-mode block changed):
%   my_code/catie_calibration/matlab/CATIE_FIXED/
%     COMPETITION_CATIE_schedule_choice_probability_hetro_FIXED.m
%     COMPETITION_CATIE_schedule_choice_probability_FIXED.m
% Shared, unmodified helpers:
%   Data_resources/competition_analysis-main/CATIE/CATIE_implementation_helpers/
%     base2dec.m, getExploreProb.m
%
% As an independent sanity check, for the three EDA schedules (4, 5, 7) the
% original model's output is compared against the previously-generated
% my_code/EDA_set/processing/eda_with_catie_probabilities.csv -- both were
% produced by the same original .m files, so they should agree to floating
% point precision regardless of anything done in this script or in Python.
%
% Run (from anywhere, non-interactively):
%   matlab -batch "run('my_code/catie_calibration/matlab/run_bug_comparison_all_schedules.m')"

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
fixed_dir    = fullfile(this_dir, 'CATIE_FIXED');
results_dir  = fullfile(this_dir, 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

addpath(fullfile(original_dir, 'CATIE_implementation_helpers'));  % base2dec, getExploreProb (shared, unmodified)
addpath(original_dir);   % COMPETITION_CATIE_schedule_choice_probability(_hetro)
addpath(fixed_dir);      % ..._FIXED variants

fprintf('project root : %s\n', project_root);
fprintf('original CATIE : %s\n', original_dir);
fprintf('fixed CATIE    : %s\n\n', fixed_dir);

%% ── 2. Schedule directories ──────────────────────────────────────────────
% {schedule_label, directory, layout}  layout: 'nested' has schedule_N as a
% subfolder of the given directory; 'flat' has CSVs directly in it.

schedules = {
    'schedule_0',  fullfile(project_root, 'my_code', 'schedule_0'),                      'flat'
    'schedule_1',  fullfile(project_root, 'my_code', 'Test_set', 'schedule_1'),          'flat'
    'schedule_2',  fullfile(project_root, 'my_code', 'Training_set', 'schedule_2'),      'flat'
    'schedule_3',  fullfile(project_root, 'my_code', 'Training_set', 'schedule_3'),      'flat'
    'schedule_4',  fullfile(project_root, 'my_code', 'EDA_set', 'schedule_4'),           'flat'
    'schedule_5',  fullfile(project_root, 'my_code', 'EDA_set', 'schedule_5'),           'flat'
    'schedule_6',  fullfile(project_root, 'my_code', 'Training_set', 'schedule_6'),      'flat'
    'schedule_7',  fullfile(project_root, 'my_code', 'EDA_set', 'schedule_7'),           'flat'
    'schedule_8',  fullfile(project_root, 'my_code', 'Test_set', 'schedule_8'),          'flat'
    'schedule_9',  fullfile(project_root, 'my_code', 'Training_set', 'schedule_9'),      'flat'
    'schedule_10', fullfile(project_root, 'my_code', 'Test_set', 'schedule_10'),         'flat'
    'schedule_11', fullfile(project_root, 'my_code', 'Training_set', 'schedule_11'),     'flat'
    };

N_TRIALS = 100;
MIN_CHOICES_PER_SIDE = 5;

%% ── 3. Load stored EDA reference (for the sanity check in step 5) ───────

eda_ref_path = fullfile(project_root, 'my_code', 'EDA_set', 'processing', 'eda_with_catie_probabilities.csv');
eda_ref = readtable(eda_ref_path, 'TextType', 'string');
eda_ref.is_choice_1_ref = strcmpi(eda_ref.is_biased_choice, 'true');

%% ── 4. Process every schedule ────────────────────────────────────────────

subj_rows = {};   % accumulate per-subject result rows (cell array of structs)
n_excluded_total = 0;
max_ref_dev = 0;   % max abs deviation vs stored EDA reference (sanity check)

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

        req = {'trial_number','is_biased_choice','side_choice','biased_reward','unbiased_reward'};
        if ~all(ismember(req, d.Properties.VariableNames))
            n_excl = n_excl + 1;
            continue;
        end

        [side_counts, ~] = groupcounts(d.side_choice);
        if numel(side_counts) < 2 || min(side_counts) < MIN_CHOICES_PER_SIDE
            n_excl = n_excl + 1;
            continue;
        end

        if height(d) ~= N_TRIALS || ~isequal(sort(d.trial_number), (0:N_TRIALS-1)')
            n_excl = n_excl + 1;
            continue;
        end

        [~, ord] = sort(d.trial_number);
        d = d(ord, :);

        rewards_1   = double(d.biased_reward);
        rewards_2   = double(d.unbiased_reward);
        is_choice_1 = strcmpi(d.is_biased_choice, 'true');

        % ── run both models (real MATLAB, both paths) ────────────────────
        % NOTE: COMPETITION_CATIE_schedule_choice_probability(_hetro) already
        % returns P(the choice actually made) -- the inner per-k function
        % applies "if is_choice_1(trial): p else 1-p" itself (see its final
        % lines). Do NOT apply that conversion again here; an earlier version
        % of this script did, which silently flipped ~40% of trials (every
        % trial where the participant chose the unbiased alternative) and
        % was caught by the sanity check in step 5 (deviation ~0.9 instead
        % of ~1e-15 against the stored EDA reference).
        pc_pub = COMPETITION_CATIE_schedule_choice_probability_hetro(rewards_1, rewards_2, is_choice_1);
        pc_fix = COMPETITION_CATIE_schedule_choice_probability_hetro_FIXED(rewards_1, rewards_2, is_choice_1);

        % ── sanity check: schedules 4/5/7 vs the stored EDA reference ─────
        if ismember(sched_label, {'schedule_4','schedule_5','schedule_7'})
            ref_rows = eda_ref(eda_ref.subject_file == string(fname) & eda_ref.schedule == string(sched_label), :);
            if height(ref_rows) == N_TRIALS
                [~, ro] = sort(ref_rows.trial_number);
                ref_rows = ref_rows(ro, :);
                dev = max(abs(pc_pub - ref_rows.catie_choice_probability));
                max_ref_dev = max(max_ref_dev, dev);
            end
        end

        n_kept = n_kept + 1;
        row = struct();
        row.schedule    = sched_label;
        row.subject_id  = sprintf('%s/%s', sched_label, fname);
        row.n_trials    = N_TRIALS;
        row.Ep_pub      = mean(pc_pub);
        row.Ep_fix      = mean(pc_fix);
        row.Elogp_pub   = mean(log(pc_pub));
        row.Elogp_fix   = mean(log(pc_fix));
        subj_rows{end+1} = row; %#ok<AGROW>
    end

    fprintf('  kept %d, excluded %d\n', n_kept, n_excl);
    n_excluded_total = n_excluded_total + n_excl;
end

fprintf('\ntotal subjects kept: %d   total excluded: %d\n', numel(subj_rows), n_excluded_total);

%% ── 5. Sanity check vs stored EDA reference ──────────────────────────────

fprintf('\n%s\n', repmat('=', 1, 78));
fprintf('SANITY CHECK: original-model output vs stored eda_with_catie_probabilities.csv\n');
fprintf('%s\n', repmat('=', 1, 78));
fprintf('max |deviation| (schedules 4/5/7 only) = %.3e  (expect ~1e-15; both were produced\n', max_ref_dev);
fprintf('by the same unmodified original .m files, so this checks nothing but I/O fidelity)\n');

%% ── 6. Assemble results table ────────────────────────────────────────────

T = struct2table([subj_rows{:}]);
writetable(T, fullfile(results_dir, 'matlab_per_subject_metrics.csv'));

%% ── 7. Pooled and per-schedule summary ───────────────────────────────────

fprintf('\n%s\n', repmat('=', 1, 78));
fprintf('POOLED, ALL SCHEDULES (real MATLAB, both original and fixed .m files)\n');
fprintf('%s\n', repmat('=', 1, 78));
fprintf('subjects %d   trials %d   schedules %d\n', height(T), height(T)*N_TRIALS, numel(unique(T.schedule)));
fprintf('E[p]      published %.4f   fixed %.4f   delta %+.4f\n', mean(T.Ep_pub), mean(T.Ep_fix), mean(T.Ep_fix)-mean(T.Ep_pub));
fprintf('E[log p]  published %.4f   fixed %.4f   delta %+.4f\n', mean(T.Elogp_pub), mean(T.Elogp_fix), mean(T.Elogp_fix)-mean(T.Elogp_pub));

% Paired t-test across subjects, implemented from scratch: this machine has
% no Statistics and Machine Learning Toolbox (ttest/tinv unavailable), but
% betainc/betaincinv are base MATLAB, and the two-sided Student-t tail
% probability has the closed form P(|T|>|t|) = betainc(df/(df+t^2), df/2, 1/2).
[t_ep, df_ep, p_ep, ci_ep]     = paired_ttest_manual(T.Ep_fix, T.Ep_pub);
[t_elp, df_elp, p_elp, ci_elp] = paired_ttest_manual(T.Elogp_fix, T.Elogp_pub);
fprintf('\npaired t-test, fixed - published, clustered at the subject level (n=%d):\n', height(T));
fprintf('  E[p]      diff %+.4f   95%% CI [%+.4f, %+.4f]   t(%d)=%.2f   p=%.3e\n', ...
    mean(T.Ep_fix - T.Ep_pub), ci_ep(1), ci_ep(2), df_ep, t_ep, p_ep);
fprintf('  E[log p]  diff %+.4f   95%% CI [%+.4f, %+.4f]   t(%d)=%.2f   p=%.3e\n', ...
    mean(T.Elogp_fix - T.Elogp_pub), ci_elp(1), ci_elp(2), df_elp, t_elp, p_elp);

fprintf('\nreference (Dan, Plonsky & Loewenstein 2025, Tables S1/S2, pooled over all\n');
fprintf('12 schedules): CATIE E[p]=0.619, E[log p]=-0.678; best QL E[log p]=-0.569.\n');

fprintf('\n%s\n', repmat('=', 1, 78));
fprintf('PER-SCHEDULE BREAKDOWN\n');
fprintf('%s\n', repmat('=', 1, 78));
fprintf('%-14s%6s%10s%10s%9s%12s%12s%9s\n', 'schedule', 'subj', 'Ep_pub', 'Ep_fix', 'delta', 'Elogp_pub', 'Elogp_fix', 'delta');

sched_names = unique(T.schedule, 'stable');
% sort by trailing number for readability
nums = cellfun(@(s) str2double(regexp(s, '\d+', 'match', 'once')), sched_names);
[~, ord] = sort(nums);
sched_names = sched_names(ord);

sched_summary = table();
for i = 1:numel(sched_names)
    g = T(strcmp(T.schedule, sched_names{i}), :);
    ep_p = mean(g.Ep_pub); ep_f = mean(g.Ep_fix);
    el_p = mean(g.Elogp_pub); el_f = mean(g.Elogp_fix);
    fprintf('%-14s%6d%10.4f%10.4f%+9.4f%12.4f%12.4f%+9.4f\n', ...
        sched_names{i}, height(g), ep_p, ep_f, ep_f-ep_p, el_p, el_f, el_f-el_p);
    sched_summary = [sched_summary; table(sched_names(i), height(g), ep_p, ep_f, el_p, el_f, ...
        'VariableNames', {'schedule','n_subjects','Ep_pub','Ep_fix','Elogp_pub','Elogp_fix'})]; %#ok<AGROW>
end
writetable(sched_summary, fullfile(results_dir, 'matlab_per_schedule_metrics.csv'));

fprintf('\nwrote:\n  %s\n  %s\n', ...
    fullfile(results_dir, 'matlab_per_subject_metrics.csv'), ...
    fullfile(results_dir, 'matlab_per_schedule_metrics.csv'));
fprintf('\ndone.\n');

%% ── local functions ──────────────────────────────────────────────────────

function [t, df, p, ci] = paired_ttest_manual(a, b)
%PAIRED_TTEST_MANUAL  Paired t-test using only base MATLAB (no toolboxes).
%   Two-sided p-value via the exact identity for the Student-t tail:
%     P(|T_df| > |t|) = betainc(df/(df+t^2), df/2, 1/2)
%   95%% CI via the inverse of the same identity (betaincinv), i.e. the
%   effective critical t-value at alpha=0.05 for this df -- not a fixed
%   z=1.96 normal-approximation shortcut, even though df is large here.
    d = a - b;
    n = numel(d);
    df = n - 1;
    m = mean(d);
    se = std(d) / sqrt(n);
    t = m / se;
    p = betainc(df / (df + t^2), df / 2, 0.5);
    x = betaincinv(0.05, df / 2, 0.5);
    tcrit = sqrt(df * (1 - x) / x);
    ci = [m - tcrit * se, m + tcrit * se];
end
