function optimize_schedule()
% optimize_schedule: Uses a genetic algorithm to optimize the rewards
% of two agents playing a game modeled as a Q-learning problem.
%
% INPUTS (via GUI):
% - OPTIMIZATION_TRIALS: Number of optimization trials to perform (default: 500)
% - GENERATION_SIZE: Size of the generation (default: 10)
% - selected_function: Function to use for optimization (default: schedule_1_CATIE)
% - save_interval: How often to save the results (default: 20)
% - plot_progress: Whether to plot progress or not (default: true)
%
% OUTPUTS:
% - optimized_1: Optimized rewards for bias-target alternative
% - optimized_2: Optimized rewards for ANTI bias-target alternative


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OPTIMIZATION_TRIALS = 100;
GENERATION_SIZE = 5;
plot_progress = true;
save_interval = 20;
schedule_save_path = 'schedule_optimization';

% Function options
function_options = {'schedule_1_CATIE', 'QL_schedule_6', 'QL_schedule_7', 'QL_schedule_9', 'QL_schedule_10'};
selected_function = @schedule_1_CATIE;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define optimization parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display and modify default values using GUI
modify_parameters();

    function modify_parameters()
        f = figure( 'Name', 'Set Parameters');

        % Explanation text
        uicontrol('Style', 'text', 'Position', [20, 400, 400, 20], 'String', 'Please set the parameters for the optimization process:');

        % OPTIMIZATION_TRIALS
        uicontrol('Style', 'text', 'Position', [20, 370, 400, 20], 'String', 'Optimization Trials (Number of optimization iterations):');
        hOptTrials = uicontrol('Style', 'edit', 'Position', [200, 350, 150, 20], 'String', num2str(OPTIMIZATION_TRIALS));
        uicontrol('Style', 'pushbutton', 'Position', [360, 350, 20, 20], 'String', '+', 'Callback', @(~,~) set(hOptTrials, 'String', num2str(str2double(get(hOptTrials, 'String')) + 1)));
        uicontrol('Style', 'pushbutton', 'Position', [380, 350, 20, 20], 'String', '-', 'Callback', @(~,~) set(hOptTrials, 'String', num2str(str2double(get(hOptTrials, 'String')) - 1)));

        % GENERATION_SIZE
        uicontrol('Style', 'text', 'Position', [20, 320, 400, 20], 'String', 'Generation Size (Number of individuals per generation):');
        hGenSize = uicontrol('Style', 'edit', 'Position', [200, 300, 150, 20], 'String', num2str(GENERATION_SIZE));
        uicontrol('Style', 'pushbutton', 'Position', [360, 300, 20, 20], 'String', '+', 'Callback', @(~,~) set(hGenSize, 'String', num2str(str2double(get(hGenSize, 'String')) + 1)));
        uicontrol('Style', 'pushbutton', 'Position', [380, 300, 20, 20], 'String', '-', 'Callback', @(~,~) set(hGenSize, 'String', num2str(str2double(get(hGenSize, 'String')) - 1)));

        % Plot progress
        uicontrol('Style', 'text', 'Position', [20, 270, 400, 20], 'String', 'Plot Progress (Display progress of optimization):');
        hPlot = uicontrol('Style', 'checkbox', 'Position', [200, 250, 20, 20], 'Value', plot_progress);

        % Save intervalclose 
        uicontrol('Style', 'text', 'Position', [20, 220, 400, 20], 'String', 'Save Interval (Frequency of saving results):');
        hSaveInt = uicontrol('Style', 'edit', 'Position', [200, 200, 150, 20], 'String', num2str(save_interval));
        uicontrol('Style', 'pushbutton', 'Position', [360, 200, 20, 20], 'String', '+', 'Callback', @(~,~) set(hSaveInt, 'String', num2str(str2double(get(hSaveInt, 'String')) + 1)));
        uicontrol('Style', 'pushbutton', 'Position', [380, 200, 20, 20], 'String', '-', 'Callback', @(~,~) set(hSaveInt, 'String', num2str(str2double(get(hSaveInt, 'String')) - 1)));

        % Function selection
        uicontrol('Style', 'text', 'Position', [20, 170, 400, 20], 'String', 'Function to use (Select optimization function):');
        hFunction = uicontrol('Style', 'popupmenu', 'Position', [200, 150, 150, 20], 'String', function_options);

        % Go button
        uicontrol('Style', 'pushbutton', 'Position', [180, 50, 150, 20], 'String', 'Start Optimization', 'Callback', @update_parameters);

        uiwait(f);

        function update_parameters(~, ~)
            OPTIMIZATION_TRIALS = str2double(get(hOptTrials, 'String'));
            GENERATION_SIZE = str2double(get(hGenSize, 'String'));
            plot_progress = logical(get(hPlot, 'Value'));
            save_interval = str2double(get(hSaveInt, 'String'));
            selected_function = str2func(function_options{get(hFunction, 'Value')});
            close(f);
        end
    end

    function mutated_generation = mutate_generation(best_schedule, GENERATION_SIZE)
        % mutate_generation - generates a mutated generation of schedules from a given best schedule
        %
        % Syntax:
        %   mutated_generation = mutate_generation(best_schedule, GENERATION_SIZE)
        %
        % Inputs:
        %   best_schedule - a row vector representing the best schedule found so far
        %   GENERATION_SIZE - the number of mutated schedules to generate
        %
        % Outputs:
        %   mutated_generation - a matrix where each row represents a mutated schedule
        %

        % Set parameters
        n = length(best_schedule); % n is the number of trials per experiment
        N_ELEMENTS_RESHUFFLE = 2; % number of elements to reshuffle to generate a new schedule

        % Initialize output
        mutated_generation = zeros(GENERATION_SIZE, n);

        % Generate mutated schedules
        for ii = 1:GENERATION_SIZE-1
            % Randomly select N_ELEMENTS_RESHUFFLE indices to reshuffle
            reshuffle_indices = randperm(n, N_ELEMENTS_RESHUFFLE);

            % If reshuffling the selected elements does not generate a new schedule, select new indices
            while isequal(best_schedule(reshuffle_indices), best_schedule(circshift(reshuffle_indices,1)))
                reshuffle_indices = randperm(n, N_ELEMENTS_RESHUFFLE);
            end

            % Mutate the best schedule by reshuffling the selected indices
            mutate_best_schedule = best_schedule;
            mutate_best_schedule(reshuffle_indices) = mutate_best_schedule(circshift(reshuffle_indices, 1));
            mutated_generation(ii, :) = mutate_best_schedule;
        end

        % Add the best schedule to the end of the mutated generation
        mutated_generation(end, :) = best_schedule;
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sem = @(v) std(v) ./ sqrt(length(v) - 1);
intial_rewards_1 = [ones(1, 25), zeros(1, 75)];
intial_rewards_2 = [zeros(1, 75), ones(1, 25)];

generation_rewards_1 = repmat(intial_rewards_1, GENERATION_SIZE, 1);
generation_rewards_2 = repmat(intial_rewards_2, GENERATION_SIZE, 1);
schedule_repetitions_base = 5;
max_scores = zeros(1, OPTIMIZATION_TRIALS);
sem_scores = zeros(1, OPTIMIZATION_TRIALS);

for optimization_repetition = 1:OPTIMIZATION_TRIALS
    schedule_repetitions = schedule_repetitions_base + 5 * optimization_repetition;
    generation_mean_biases = zeros(1, GENERATION_SIZE);
    generation_sem_biases = zeros(1, GENERATION_SIZE);

    for generation_id = 1:GENERATION_SIZE
        generation_biases = zeros(1, schedule_repetitions);
        for schedule_score_iteration = 1:schedule_repetitions
            generation_biases(schedule_score_iteration) = selected_function(generation_rewards_1(generation_id, :), generation_rewards_2(generation_id, :));
        end
        generation_mean_biases(generation_id) = mean(generation_biases);
        generation_sem_biases(generation_id) = sem(generation_biases);
    end

    [max_scores(optimization_repetition), best_index] = max(generation_mean_biases);
    sem_scores(optimization_repetition) = generation_sem_biases(best_index);

    % Plot the progress if required
    if plot_progress
        errorbar(1:optimization_repetition, max_scores(1:optimization_repetition), sem_scores(1:optimization_repetition));
        xlabel('Trial');
        ylabel('Maximal Bias');
        drawnow;
    end

    % Generate next generation
    optimized_1 = generation_rewards_1(best_index, :);
    optimized_2 = generation_rewards_2(best_index, :);
    generation_rewards_1 = mutate_generation(optimized_1, GENERATION_SIZE);
    generation_rewards_2 = mutate_generation(optimized_2, GENERATION_SIZE);

    % Save the best schedule so far every save_interval iterations
    if mod(optimization_repetition, save_interval) == 0
        current_date = datestr(now, 'yyyy-mm-dd');
        filename = sprintf('%s_%s.mat', schedule_save_path, current_date);
        save(filename, 'optimized_1', 'optimized_2');
    end
end

end