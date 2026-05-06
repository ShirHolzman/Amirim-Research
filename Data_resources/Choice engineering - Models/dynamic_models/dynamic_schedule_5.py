###############################################################################
# Code start
###############################################################################
import sys
import ast
import random
TOTAL_REWARDS = 25
NUMBER_OF_TRIALS = 100
REWARD = 1
NO_REWARD = 0


def allocate(target_allocations, anti_target_allocations, is_target_choices):
    # check if this is the first target choice, if so - reward it
    if sum(is_target_choices) == 0:
        target_alternative, anti_target_alternative = REWARD, NO_REWARD
        return target_alternative, anti_target_alternative


    # check k pattern

    # first 5 rewards
    if sum(target_allocations) + sum(anti_target_allocations) < 6:
        k = 2

    # next 20 rewards
    else:
        k = 4

    # check if participant completed the pattern
    has_pattern = True
    pattern_list = is_target_choices[-1*(k-1):]
    for target in pattern_list:
        if not target:
            has_pattern = False
            break

    target_alternative = NO_REWARD
    if has_pattern:
        target_alternative = REWARD

    anti_target_alternative = NO_REWARD

    if len(is_target_choices) > 75:
        anti_target_alternative = REWARD

    return constrain(target_allocations, target_alternative),\
           constrain(anti_target_allocations, anti_target_alternative)

def constrain(previous_allocation, current_allocation):
    """
    Constrain the current allocation based on previous allocations, such that
     both (1) no more than 25 rewards are allocated and (2) assuring that all
     25 rewards are indeed allocated.
    :param previous_allocation:
    :param current_allocation:
    :return: A constrained allocation
    """
    allocated_rewards = sum(previous_allocation)

    # If all rewards were already allocated, no more rewards may be allocated
    if allocated_rewards>=TOTAL_REWARDS:
        return 0

    # If there are as many trials left as rewards left, in all remaining trials
    # rewards should be allocated
    current_trial_number = len(previous_allocation)
    remaining_trials = NUMBER_OF_TRIALS - current_trial_number
    remaining_rewards = allocated_rewards
    if remaining_trials == (TOTAL_REWARDS-allocated_rewards):
        return 1

    # No constrain should be imposed
    return current_allocation

###############################################################################
# Template Infrastructure - Do not change
###############################################################################
REWARDS_BOTH_ALTERNATIVES = '1, 1'
REWARD_TARGET_ONLY = '1, 0'
REWARD_ANTI_TARGET_ONLY = '0, 1'
NO_REWARDS_BOTH_ALTERNATIVES = '0, 0'

def parse_input():
    """
    Get the command-line parameters with which this script is initiated (as
    explained in the script's intro, see "How should a dynamic allocation model
    file be used")
    :return: A tuple with previous
        (rewards allocation to target alternative: {1=reward, 0=no reward},
        rewards allocation to anti-target alternative: {1=reward, 0=no reward},
        choices: {1=choice in target alternative, 0=choice in anti-target})
    """
    if sys.argv[1]=="[]": #This is first trial, don't try parsing:
        return [], [], []
    else:
        target_allocations = parse_lst(sys.argv[1])
        anti_target_allocations = parse_lst(sys.argv[2])
        is_target_choices = parse_lst(sys.argv[3])
        return target_allocations, anti_target_allocations, is_target_choices



def output(target, anti_target):
    """
    Output the allocation of rewards for next trial by printing them to
     standard output.
     NOTE: It is the reward allocator (i.e. your) responsibility to enforce the
            constraint of exactly 25 allocations per alternatives (it will
            otherwise be bluntly enforced and may alter your allocations).
    :param target: A boolean indicator of reward to the alternative in which
                    *maximal* choice should be induced.
                    True indicate an allocation of reward in next trial and
                    false indicate no reward allocation.
    :param anti_target: A boolean indicator of reward to the alternative in
                    which *minimal* choice should be induced.
                    True indicate an allocation of reward in next trial and
                    false indicate no reward allocation.

    :return: None
    """
    if target and anti_target:
        print(REWARDS_BOTH_ALTERNATIVES)
    elif target and not anti_target:
        print(REWARD_TARGET_ONLY)
    elif not target and anti_target:
        print(REWARD_ANTI_TARGET_ONLY)
    elif not target and not anti_target:
        print(NO_REWARDS_BOTH_ALTERNATIVES)

def parse_lst(lst):
    as_python_lst = lst.strip('[').strip(']').split(',')
    # as_python_elements = [ast.literal_eval(el) for el in as_python_lst]
    as_python_elements = eval(lst)
    return as_python_elements

###############################################################################
# Run
###############################################################################


def simulate_experiment():
    """
    Function for testing our allocation strategy
    :return:
    """
    # init
    target_allocations = []
    anti_target_allocations = []
    is_target_choices = []
    rewards = []

    # run trials
    for i in range(NUMBER_OF_TRIALS):
        allocation = allocate(target_allocations, anti_target_allocations, is_target_choices)
        target_allocations.append(allocation[0])
        anti_target_allocations.append(allocation[1])
        if random.random() < 0.5:
            choice = 0
        else:
            choice = 1
        is_target_choices.append(choice)

        reward = 0
        if (choice == 1) & (allocation[0] == 1):
            reward = 1
        elif (choice == 0) & (allocation[1] == 1):
            reward = 1
        rewards.append(reward)

    # print(is_target_choices)
    # print(target_allocations)
    # print(sum(target_allocations))
    # print(anti_target_allocations)
    # print(sum(anti_target_allocations))
    # print(rewards)



if __name__ == '__main__':
    # original main code with system arguments

    input = parse_input()

    target_allocations = input[0]
    anti_target_allocations = input[1]
    is_target_choices = input[2]

    allocation = allocate(*input)
    allocation = (constrain(target_allocations, allocation[0]),
                  constrain(anti_target_allocations, allocation[1]))


    # simulate_experiment()
    # exit()
    # modified code for debugging
    # target_allocations = [1,1,1,1,1,1]
    # anti_target_allocations = [0,0,0,0,0,0]
    # is_target_choices = [1,0,0,1,1,0,0,1,1,0,1,1]
    # allocation = allocate(target_allocations, anti_target_allocations, is_target_choices)

    output(*allocation)

