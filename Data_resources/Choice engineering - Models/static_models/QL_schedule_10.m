% This function implements the Q-learning algorithm for two alternative
% forced choice tasks with constant reward probabilities. It evaluates
% the bias towards the first alternative for a given pair of reward
% probabilities.
%
% INPUTS:
%   - rewards_1: a vector of length N that defines the reward probability for
%       the first alternative on each trial. It should contain only 0's or 1's.
%   - rewards_2: a vector of length N that defines the reward probability for
%       the second alternative on each trial. It should contain only 0's or 1's.
%   - ETA: a scalar that defines the learning rate. It should be between 0 and 1.
%   - BETA: a scalar that defines the inverse temperature. It should be positive.
%   - EPSILON: a scalar that defines the probability of choosing a random action
%       instead of the optimal one. It should be between 0 and 1.
%   - REPETITIONS: a scalar that defines the number of repetitions. It should be
%       a positive integer.
%
% OUTPUTS:
%   - biases: a vector of length REPETITIONS that contains the bias towards
%       the first alternative for each repetition.

function alternative_1_choices = schedule_10(rewards_1, rewards_2)

BETA_MAX = 500; % Maximum allowed beta value
    function constrained_beta = constrain_beta(beta)
        constrained_beta = beta;
        constrained_beta(beta >= BETA_MAX) = BETA_MAX;
    end

% Heterogeneous online model parameters
% Heterogeneous lab model parameters
BETA_CONVERSION_FACTOR = 257/52; % Convert beta value by ratio of std dev of rewards of lab and online datasets
ETA = [1.0, 1.0, 0.46416, 1.0, 0.1, 1.0, 0.46416, 0.01, 1.0, 0.21544, 1.0, 0.21544, 1.0, 0.46416, 0.001, 0.046416, 0.21544, 0.1, 0.46416, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 0.21544, 0.21544, 0.001, 0.46416, 1.0, 0.21544, 1.0, 1.0, 0.046416, 1.0, 0.46416, 1.0, 0.046416, 0.1, 0.46416, 0.46416, 0.001, 0.46416, 1.0, 0.21544, 1.0, 1.0, 0.46416, 1.0, 0.1, 0.01, 1.0, 1.0, 1.0, 0.1, 1.0, 0.46416, 1.0, 0.21544, 0.1, 0.21544, 1.0, 0.001, 0.001, 1.0, 0.1, 0.046416, 1.0, 0.046416, 0.1, 1.0, 0.046416, 0.021544, 1.0, 0.46416, 1.0, 0.1, 1.0, 0.1, 0.46416, 0.21544, 0.046416, 0.46416, 0.46416, 0.21544, 0.21544, 0.1, 0.46416, 0.21544, 0.1, 1.0, 0.46416, 0.46416, 0.046416, 0.21544, 1.0, 0.21544, 1.0, 0.46416, 1.0, 0.46416, 0.46416, 1.0, 0.1, 0.001, 0.046416, 0.01, 0.001, 0.21544, 0.46416, 0.1, 1.0, 0.1, 0.21544, 0.01, 0.021544, 1.0, 0.46416, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 0.46416, 1.0, 0.021544, 0.21544, 0.046416, 0.046416, 0.21544, 1.0, 0.1, 0.001, 0.046416, 1.0, 0.46416, 1.0, 1.0, 0.1, 0.046416, 1.0, 0.021544, 0.21544, 0.46416, 0.21544, 1.0, 0.21544, 0.1, 0.1, 1.0, 0.46416, 0.21544, 1.0, 0.01, 0.046416, 1.0, 0.21544, 1.0, 0.1, 0.046416, 0.001, 0.046416, 0.021544, 0.046416, 0.21544, 0.46416, 0.21544, 0.001, 0.21544, 0.21544, 1.0, 1.0, 1.0, 0.046416, 1.0, 0.46416, 0.1, 1.0, 0.001, 1.0, 0.21544, 1.0, 1.0, 0.46416, 0.1, 0.21544, 0.21544, 0.46416, 0.21544, 1.0, 1.0, 1.0, 0.21544, 0.21544, 0.1];
BETA = BETA_CONVERSION_FACTOR*[1000.0, 1000.0, 5.9948, 21.544, 1.6681, 1000.0, 5.9948, 21.544, 1000.0, 1000.0, 77.426, 5.9948, 1000.0, 1.6681, 1000.0, 77.426, 21.544, 21.544, 77.426, 1000.0, 1000.0, 5.9948, 21.544, 5.9948, 21.544, 1000.0, 5.9948, 21.544, 0.01, 77.426, 77.426, 1.6681, 1000.0, 5.9948, 1000.0, 1000.0, 77.426, 1000.0, 5.9948, 5.9948, 0.46416, 21.544, 0.01, 77.426, 1000.0, 1000.0, 1000.0, 1000.0, 1.6681, 0.46416, 5.9948, 21.544, 1000.0, 0.46416, 0.12915, 0.46416, 1000.0, 1000.0, 1000.0, 5.9948, 1.6681, 1.6681, 1000.0, 1000.0, 5.9948, 1000.0, 278.26, 21.544, 1000.0, 1.6681, 77.426, 1000.0, 0.46416, 21.544, 1000.0, 1000.0, 1000.0, 1.6681, 1000.0, 21.544, 1000.0, 0.46416, 5.9948, 77.426, 1000.0, 1.6681, 21.544, 1.6681, 77.426, 77.426, 1.6681, 1.6681, 1.6681, 278.26, 0.46416, 1.6681, 1000.0, 0.46416, 21.544, 1000.0, 1.6681, 77.426, 77.426, 21.544, 0.46416, 0.035938, 5.9948, 278.26, 1000.0, 1.6681, 0.46416, 1000.0, 1000.0, 5.9948, 1.6681, 77.426, 1000.0, 1000.0, 21.544, 1000.0, 1000.0, 21.544, 1000.0, 1000.0, 21.544, 21.544, 77.426, 1.6681, 5.9948, 21.544, 5.9948, 1.6681, 21.544, 77.426, 5.9948, 1000.0, 5.9948, 1000.0, 5.9948, 1000.0, 21.544, 278.26, 278.26, 5.9948, 21.544, 278.26, 5.9948, 21.544, 1000.0, 0.12915, 1.6681, 21.544, 1.6681, 1.6681, 278.26, 5.9948, 21.544, 278.26, 1000.0, 21.544, 5.9948, 21.544, 1000.0, 1.6681, 5.9948, 1000.0, 1.6681, 77.426, 1000.0, 278.26, 0.46416, 1000.0, 1000.0, 1000.0, 21.544, 1000.0, 1000.0, 1000.0, 77.426, 1.6681, 77.426, 5.9948, 278.26, 0.01, 5.9948, 5.9948, 21.544, 21.544, 5.9948, 21.544, 21.544, 21.544, 0.46416, 21.544, 21.544, 5.9948, 278.26, 77.426, 1000.0, 1000.0];
BETA = constrain_beta(BETA);
EPSILON = [0.11111, 0.27778, 0.11111, 0.22222, 0.11111, 0.16667, 0.11111, 0.27778, 0.11111, 0.055556, 0.055556, 0.11111, 0.16667, 0.11111, 0.16667, 0.22222, 0.055556, 0.22222, 0.27778, 0.33333, 0.27778, 0.055556, 0.22222, 0.22222, 0.055556, 0.11111, 0.16667, 0.27778, 0.055556, 0.27778, 0.22222, 0.055556, 0.22222, 0.11111, 0.33333, 0.22222, 0.055556, 0.16667, 0.33333, 0.22222, 0.0, 0.22222, 0.5, 0.27778, 0.22222, 0.22222, 0.16667, 0.33333, 0.055556, 0.0, 0.16667, 0.11111, 0.055556, 0.0, 0.0, 0.055556, 0.11111, 0.11111, 0.22222, 0.11111, 0.055556, 0.055556, 0.11111, 0.11111, 0.16667, 0.11111, 0.16667, 0.055556, 0.11111, 0.16667, 0.11111, 0.055556, 0.0, 0.055556, 0.055556, 0.055556, 0.055556, 0.16667, 0.33333, 0.27778, 0.16667, 0.22222, 0.33333, 0.16667, 0.27778, 0.16667, 0.11111, 0.16667, 0.16667, 0.22222, 0.11111, 0.27778, 0.055556, 0.22222, 0.22222, 0.11111, 0.055556, 0.16667, 0.22222, 0.38889, 0.22222, 0.22222, 0.16667, 0.22222, 0.0, 0.0, 0.16667, 0.22222, 0.055556, 0.055556, 0.0, 0.16667, 0.11111, 0.11111, 0.055556, 0.16667, 0.33333, 0.055556, 0.22222, 0.11111, 0.22222, 0.27778, 0.11111, 0.22222, 0.38889, 0.11111, 0.16667, 0.22222, 0.22222, 0.16667, 0.22222, 0.16667, 0.22222, 0.16667, 0.22222, 0.22222, 0.055556, 0.055556, 0.11111, 0.055556, 0.16667, 0.055556, 0.11111, 0.38889, 0.27778, 0.27778, 0.22222, 0.16667, 0.27778, 0.0, 0.055556, 0.11111, 0.16667, 0.22222, 0.055556, 0.055556, 0.27778, 0.16667, 0.22222, 0.27778, 0.22222, 0.11111, 0.16667, 0.055556, 0.055556, 0.16667, 0.055556, 0.22222, 0.16667, 0.27778, 0.0, 0.38889, 0.055556, 0.055556, 0.27778, 0.33333, 0.16667, 0.055556, 0.33333, 0.16667, 0.22222, 0.16667, 0.16667, 0.5, 0.16667, 0.27778, 0.16667, 0.22222, 0.22222, 0.27778, 0.33333, 0.22222, 0.11111, 0.055556, 0.27778, 0.27778, 0.16667, 0.33333, 0.16667, 0.33333]; 


% Get the length of the rewards vectors
N = length(rewards_1);
N_params = length(ETA); % Number of "sub agents"


% Each sub agent has its own q-value
q1 = NaN(1, N_params);
q2 = NaN(1, N_params);
all_agents_p1 = NaN(1, N_params); % The probability of choosing alternative 1

% In the first two trials, the agent chooses the two alternatives in
% random order, then applies "reset of initial conditions"
if rand<0.5
    q1(:) = rewards_1(1); q2(:) = rewards_2(2);
else
    q1(:) = rewards_1(2); q2(:) = rewards_2(1);
end

% Initialize the count of choices for alternative 1 to 1
alternative_1_choices = 1;

% Run the Q-learning algorithm for each trial
for trial=3:N

    % Compute the probability of choosing alternative 1 using an epsilon-
    % softmax rule
    for agent_id=1:N_params
        all_agents_p1(agent_id) = epsilon_softmax_p_choice_1(q1(agent_id), q2(agent_id), EPSILON(agent_id), BETA(agent_id));
    end

    p1 = mean(all_agents_p1);
    % Choose an action using the computed probability
    if p1>rand()
        % The agent chose alternative 1
        alternative_1_choices = alternative_1_choices + 1;
        % Update the Q-value for alternative 1
        q1 = q1 + ETA.*(rewards_1(trial) - q1);
    else
        % The agent chose alternative 2
        % Update the Q-value for alternative 2
        q2 = q2 + ETA.*(rewards_2(trial) - q2);
    end
end
end
