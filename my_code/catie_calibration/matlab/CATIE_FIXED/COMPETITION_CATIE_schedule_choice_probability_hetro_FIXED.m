function p_decisions = COMPETITION_CATIE_schedule_choice_probability_hetro_FIXED(rewards_1, rewards_2, is_choice_1)
%COMPETITION_CATIE_SCHEDULE_CHOICE_PROBABILITY_HETRO_FIXED
%   Identical to Data_resources/competition_analysis-main/CATIE/
%   COMPETITION_CATIE_schedule_choice_probability_hetro.m except that it
%   calls COMPETITION_CATIE_schedule_choice_probability_FIXED instead of the
%   original per-k function. See that file's header for the fix itself.
%   The mixture/weighting logic (the SHIPPED time-averaged posterior, not the
%   per-trial weighting the original comment on line 26 describes) is left
%   untouched, so the only difference between this function's output and the
%   original hetro function's output is attributable to the heuristic-mode fix
%   alone. That is deliberate: this file exists to isolate the heuristic fix
%   against the shipped code, so it must NOT also change the mixing.
%   Note (2026-09): the per-trial weighting on line 26 -- not the time-average
%   this file preserves -- is what the paper's reported numbers match. So this
%   function reproduces the SHIPPED code with the heuristic fixed, which is a
%   different thing from reproducing the paper. See ../README.md section 4.

% Define the number of sub-agents and the number of decisions in the task.
K = 0:2;
n_sub_agents = length(K);
n_decisions = length(is_choice_1);

% Pre-allocate a matrix to store the choice probabilities of each sub-agent for each decision.
agents_choice_probabilities = zeros(length(K), n_decisions);

% Loop over all sub-agents and compute their choice probabilities for each decision.
for agent_index=1:n_sub_agents
    k = K(agent_index);
    agents_choice_probabilities(agent_index,:) = COMPETITION_CATIE_schedule_choice_probability_FIXED(rewards_1, rewards_2, is_choice_1, k);
end

% Compute the likelihood of each sub-agent given the observed choices of all sub-agents, for each decision.
agents_likelihoods_in_each_trial = cumprod([ones(n_sub_agents, 1), agents_choice_probabilities],2); % ones in the first elemnet to represnt the prior likelihood of each k is equal
normalized_agents_lielihoods_in_each_trial = agents_likelihoods_in_each_trial./sum(agents_likelihoods_in_each_trial);
normalized_agents_lielihoods_in_all_but_last_trial = normalized_agents_lielihoods_in_each_trial(:, 1:end-1);

% Compute the average choice probabilities across all sub-agents, weighted by their likelihoods.
p_decisions = mean(agents_choice_probabilities' * normalized_agents_lielihoods_in_all_but_last_trial,2);
% p_decisions = sum(agents_choice_probabilities .* normalized_agents_lielihoods_in_all_but_last_trial);

end
