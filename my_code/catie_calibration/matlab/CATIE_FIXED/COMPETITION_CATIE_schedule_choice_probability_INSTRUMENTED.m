function [p_decisions, H_out, b_out, c_prev_out, s_prev_out, sbar_prev_out, g_out] = ...
    COMPETITION_CATIE_schedule_choice_probability_INSTRUMENTED(rewards_1, rewards_2, is_choice_1, k)
%COMPETITION_CATIE_SCHEDULE_CHOICE_PROBABILITY_INSTRUMENTED
%   Identical model to COMPETITION_CATIE_schedule_choice_probability_FIXED.m
%   (same corrected heuristic branch), but additionally exports the per-trial
%   internal quantities that my_code/catie_calibration/catie_core.py's
%   state_tensors() is meant to reproduce. Written for exactly one purpose:
%   a true element-by-element cross-check of state_tensors() against MATLAB's
%   own loop variables, independent of the end-to-end probability comparisons
%   done elsewhere (golden_test.py, run_bug_comparison_all_schedules.m), which
%   only ever compared the FINAL choice probability.
%
%   Exported arrays (all length nTrials, 1-indexed by trial like the rest of
%   this function; index 1 = trial 1):
%
%     H_out(trial)         is_test_trend (1/0). 0 at trial 1 (untested).
%     b_out(trial)          heuristic verdict: 1 if the trend rule selects
%                           alternative 1, 0 otherwise. Only meaningful where
%                           H_out(trial)==1. 0 at trial 1.
%     c_prev_out(trial)     is_choice_1(trial-1) (1/0). 0 at trial 1.
%     s_prev_out(trial)     surprise(trial-1). 0 at trial 1.
%     sbar_prev_out(trial)  mean_surprise as of the END of trial-1 (i.e. the
%                           value used when predicting `trial`, captured
%                           before trial's own outcome updates it). 0 at
%                           trial 1.
%     g_out(trial)          p_choice_1_contingency_mode as computed for this
%                           trial. Computed (and exported) for EVERY trial,
%                           including trial 1 -- unlike the others, this
%                           block runs unconditionally, before the
%                           trial==1/else split. catie_core.py's state_tensors
%                           computes the same value at trial 1 but does NOT
%                           store it (only writes g_arr(t) inside its
%                           "if t > 1" block), because trial 1 is hardcoded to
%                           p=0.5 and never reads g. When cross-checking
%                           against Python, compare g_out(2:end) only, and
%                           treat g_out(1) as a legitimate MATLAB value with
%                           no Python counterpart to compare it to -- not a
%                           mismatch.
%
%   Every other line is copied verbatim from
%   COMPETITION_CATIE_schedule_choice_probability_FIXED.m; only the six
%   export lines (marked "% === INSTRUMENTATION ===") were added.

%% parameters

pHeuristic= 0.29;
epsilon= 0.3;
pIner= 0.71;

%% constants
nTrials = 100;

%% preallocation and initializations
contingency_counts_2 = zeros(4^k,2,'double');
contingency_counts_1 = zeros(4^k,2,'double');
pays = NaN(nTrials,1);
surprise = NaN(nTrials,1); % holds levels of surprise in each trial
totalSurprise = 0; % holds sum of surprises thus far
reward_vec = NaN(nTrials,1); % holds observed history (0 = alternative 2, no reward, 1 = alternative 2, with reward,
                                                     % 2 = alternative 1, no reward, 3 = alternative 1, with reward)
observed_SD = zeros(1,2); % holds the observed SD from payoffs in both options
reward_mean_2 = 0; % grand mean from alternative 2 (initialized to belief of zero)
reward_mean_1 = 0; % grand mean from alternative 1 (initialized to belief of zero)
expected_reward = zeros(1,2); % holds the payoff expectations for next trial in each alterrnative. Initialized by 0
reward_sum_2 = 0; % Holds sum of payoffs obtained from A
sum_reward_squared_2 = 0; %Holds sum of squares for payoffs from A
num_choices_2 = 0; % Holds number of alternative 2 choices made
reward_sum_1 = 0; % Holds sum of payoffs obtained from alternative 1
sum_reward_squared_1 = 0; %Holds sum of squares for payoffs from B
num_choices_1 = 0; % Holds number of alternative 1 choices made

p_decisions = zeros(1, nTrials);
POSITIVE_REWARD = 1;

% === INSTRUMENTATION ===
H_out         = zeros(1, nTrials);
b_out         = zeros(1, nTrials);
c_prev_out    = zeros(1, nTrials);
s_prev_out    = zeros(1, nTrials);
sbar_prev_out = zeros(1, nTrials);
g_out         = zeros(1, nTrials);
% === end instrumentation ===

%% Calculate decision probabilities
for trial = 1:nTrials
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Choice probability under contingency mode, beofre observing the
    % reward of current trial.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if k == 0 % CAB-0
        expected_reward(1) = reward_mean_1;
        expected_reward(2) = reward_mean_2;
        if reward_mean_1 == reward_mean_2
            p_choice_1_contingency_mode = 0.5;  %    IF ESTIMATED VALUES ARE EQUAL, CHOOSE RANDOMLY
        else
            p_choice_1_contingency_mode = reward_mean_1 > reward_mean_2;
        end
    elseif (trial-1) > k && trial < nTrials % CAB-k, k>0; trial-1, because in current trial reward was not observed yet
        last_k_trials_indices = (trial-k):(trial-1);
        current_k_contingency = (reward_vec(last_k_trials_indices))';
        rowDecide = base2dec(4, current_k_contingency)+1; % current contingency
        if contingency_counts_2(rowDecide,2) > 0 % If current contingency exists
           CA_2 = contingency_counts_2(rowDecide,1) / contingency_counts_2(rowDecide,2);
        else
            existing_contingencies_indices = contingency_counts_2(:,2) > 0;
            if sum(existing_contingencies_indices) > 0 % If any other contingency exists
                CA_2 = contingency_counts_2(existing_contingencies_indices, 1)./contingency_counts_2(existing_contingencies_indices, 2);
            else % No contingency exists, consider the simple reward mean
                CA_2 = reward_mean_2;
            end
        end
        if contingency_counts_1(rowDecide,2) > 0  % If current contingency exists
           CA_1 = contingency_counts_1(rowDecide,1) / contingency_counts_1(rowDecide,2);
        else
            existing_contingencies_indices = contingency_counts_1(:,2) > 0; % If any other contingency exists
            if sum(existing_contingencies_indices) > 0
                CA_1 = contingency_counts_1(existing_contingencies_indices, 1)./contingency_counts_1(existing_contingencies_indices, 2);
            else  % No contingency exists, consider the simple reward mean
                CA_1 = reward_mean_1;
            end
        end
        [CA_1_values, CA_2_values] = meshgrid(CA_1, CA_2);
        mean_contingencies_1_greater_than_2 = mean(CA_1_values(:)>CA_2_values(:));
        mean_contingencies_1_equal_2 = mean(CA_1_values(:)==CA_2_values(:));
        p_choice_1_contingency_mode = mean_contingencies_1_greater_than_2 + 0.5*mean_contingencies_1_equal_2;

        expected_reward(2) = mean(CA_2);
        expected_reward(1) = mean(CA_1);

    else %if k>0 and not enough experience to use CAB-k
        if reward_mean_1 == reward_mean_2
            p_choice_1_contingency_mode = 0.5;
        else
            p_choice_1_contingency_mode = reward_mean_1 > reward_mean_2;
        end
    end

    % === INSTRUMENTATION: g is computed unconditionally above, every trial ===
    g_out(trial) = p_choice_1_contingency_mode;
    % === end instrumentation ===

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculate decision probability in current trial
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if trial==1
        p_decisions(1) = 0.5;
        % H_out, b_out, c_prev_out, s_prev_out, sbar_prev_out stay at their
        % preallocated 0 for trial 1, matching catie_core.py's convention.
    else
        % Heuristic mode (FIXED, see COMPETITION_CATIE_schedule_choice_probability_FIXED.m
        % header for the derivation): compare pays(trial-1) vs pays(trial-2),
        % use is_choice_1(trial-1) throughout.
        is_test_trend = (trial>2) && (is_choice_1(trial-1) == is_choice_1(trial-2)) && ((pays(trial-1) ~= pays(trial-2)));
        % === INSTRUMENTATION ===
        H_out(trial) = double(is_test_trend);
        % === end instrumentation ===
        if is_test_trend
            if ((is_choice_1(trial-1) && (pays(trial-1) > pays(trial-2))) ||... % Last choice in 1 and positive trend
                    (~is_choice_1(trial-1) && (pays(trial-1) < pays(trial-2))))   % Last choice in 2 and negative trend
                p_choose_1_heuristic_mode = pHeuristic;
                b_out(trial) = 1; % === INSTRUMENTATION ===
            else
                p_choose_1_heuristic_mode = 0;
                b_out(trial) = 0; % === INSTRUMENTATION ===
            end
            p_try_explore = 1-pHeuristic;
        else
            p_try_explore = 1;
            p_choose_1_heuristic_mode = 0;
            b_out(trial) = 0; % === INSTRUMENTATION ===
        end

        % exploration mode
        p_explore = getExploreProb(epsilon, surprise(trial-1), mean_surprise);
        p_enter_explore_mode = p_try_explore * p_explore;
        p_choose_1_exploration_mode = 0.5*p_enter_explore_mode;

        % Inertia mode
        p_try_inertia_mode = p_try_explore * (1-p_explore);
        p_enter_inertia_mode = p_try_inertia_mode*pIner;
        p_choose_1_inertia_mode = is_choice_1(trial-1)*p_enter_inertia_mode;

        % === INSTRUMENTATION: capture the three "previous-trial" inputs at
        % exactly the point they are read, before this trial's own outcome
        % overwrites surprise(trial) / mean_surprise below ===
        c_prev_out(trial)    = double(is_choice_1(trial-1));
        s_prev_out(trial)    = surprise(trial-1);
        sbar_prev_out(trial) = mean_surprise;
        % === end instrumentation ===

        % Contingent average/exploitation mode
        p_enter_ca_mode = p_try_inertia_mode*(1-pIner);
        p_choose_1_ca_mode = p_enter_ca_mode*p_choice_1_contingency_mode;

        current_p_choice_1 = p_choose_1_heuristic_mode + p_choose_1_exploration_mode + p_choose_1_inertia_mode + p_choose_1_ca_mode;
        if is_choice_1(trial)
            p_decisions(trial) = current_p_choice_1;
        else
            p_decisions(trial) = 1-current_p_choice_1;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Update internal parameters given curren choice and its observed
    % outcome
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if is_choice_1(trial) % Chose alternative 1
        pays(trial) = rewards_1(trial);
        if pays(trial) == POSITIVE_REWARD
            reward_vec(trial) = 3;
        else
            reward_vec(trial) = 2;
        end
        reward_sum_1 = reward_sum_1 + pays(trial);
        num_choices_1 = num_choices_1 + 1;
        sum_reward_squared_1 = sum_reward_squared_1 + (pays(trial)).^2;
        var_reward_1 = (1/(num_choices_1-1)) * (sum_reward_squared_1 - ((reward_sum_1^2)/num_choices_1));
        if isnan(var_reward_1) || var_reward_1 < 0
            var_reward_1 = 0;
        end
        observed_SD(1) = sqrt(var_reward_1);
    else % Chose alternative 2
        pays(trial) = rewards_2(trial);
        if pays(trial) == POSITIVE_REWARD
            reward_vec(trial) = 1;
        else
            reward_vec(trial) = 0;
        end
        reward_sum_2 = reward_sum_2 + pays(trial);
        num_choices_2 = num_choices_2 + 1;
        sum_reward_squared_2 = sum_reward_squared_2 + (pays(trial)).^2;
        var_reward_2 = (1/(num_choices_2-1)) * (sum_reward_squared_2 - ((reward_sum_2^2)/num_choices_2));
        if isnan(var_reward_2) || var_reward_2 < 0
            var_reward_2 = 0;
        end
        observed_SD(2) = sqrt(var_reward_2);
    end

    % update surprise resulting from observed payoff
    chosen_alternative = (1-is_choice_1(trial)) + 1; % 1-->1; 0-->2
    if observed_SD(chosen_alternative) > 0.0001  % SURPRISE FUNCTION (DEPENDENT ON EXPECTATIONS & SDs)
        expected_vs_received_reward_abs_diff = abs(expected_reward(chosen_alternative) - pays(trial));
        surprise(trial) =  expected_vs_received_reward_abs_diff/(observed_SD(chosen_alternative) + expected_vs_received_reward_abs_diff); % surprise function
    else % no variability observed in the chosen button
        surprise(trial) = 0;
    end
    totalSurprise = totalSurprise + surprise(trial);
    mean_surprise = totalSurprise / trial; % the mean surprise observed in this game

    % update contingency beliefs
    if k > 0 && trial > k
        rowUpdate = base2dec(4,(reward_vec((trial-k):(trial-1))'))+1;
        if is_choice_1(trial)
            contingency_counts_1(rowUpdate,1) = contingency_counts_1(rowUpdate,1) + pays(trial);
            contingency_counts_1(rowUpdate,2) = contingency_counts_1(rowUpdate,2) + 1;
        else
            contingency_counts_2(rowUpdate,1) = contingency_counts_2(rowUpdate,1) + pays(trial);
            contingency_counts_2(rowUpdate,2) = contingency_counts_2(rowUpdate,2) + 1;
        end
    end

    if is_choice_1(trial) % update General Mean
        reward_mean_1 = reward_sum_1/num_choices_1;
    else
        reward_mean_2 = reward_sum_2/num_choices_2;
    end

end
end
