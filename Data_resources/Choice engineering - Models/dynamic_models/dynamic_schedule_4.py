import sys
import ast
TOTAL_REWARDS = 25
NUMBER_OF_TRIALS = 100
REWARD = 1
NO_REWARD = 0

###############################################################################
# Template - Insert your code here
###############################################################################
import random
count1 = 0
count2 = 0

def addition (is_target_choices, target_allocations, anti_target_allocations):
    global count1
    global count2
    length = min(len(is_target_choices), len(target_allocations), len(target_allocations))
    output = (0, 1)
    counter = 0
    counter_target_true = 0
    counter_target_false = 0
    counter_anti_target_true = 0
    counter_anti_target_false = 0
    for idx in range(length):
        if is_target_choices[idx] == 1:
            counter += 1
    for idx_target in range(len(target_allocations)):
        if target_allocations[idx_target] == 1:
            counter_target_true += 1
        else:
            counter_target_false += 1

    for idx_anti in range(len(anti_target_allocations)):
        if anti_target_allocations[idx_anti] == 1:
            counter_anti_target_true += 1
        else:
            counter_anti_target_false += 1

    if counter < 2:
        output = (1, 0)
        return output

    elif count1 < 9:
        rand = int(random.uniform(0, 2))
        count1 += 1
        output = (rand, 0)
        return output
    elif count2 < 12:
        rand = int(random.uniform(0, 3))
        if rand == 2:
            rand = 0
        count2 += 1
        output = (rand, 0)
        return output

    elif count2 == 12:
        counter_last_8 = 0
        rand = int(random.uniform(0, 2))
        count2 += 1
        for idx_last_8 in range(max(0, length-8), length):
            if is_target_choices[idx_last_8] == 1:
                counter_last_8 += 1
        if counter_last_8 > 3:
            count2 = 8
        output = (rand, 0)
        return output

    elif is_target_choices[length - 1] == 0:
        rand = int(random.uniform(0, 3))
        if rand == 2:
            rand = -1
        count1 = rand + 7
        count2 = 8
        output = (1, 0)
        return output

    elif counter_anti_target_true > 24 and counter_target_true > 24:
        output = (0, 0)
        return output
    elif counter_target_true > 24 and counter_anti_target_false > 74:
        output = (0, 1)
        return output
    elif counter_target_false > 74 and counter_anti_target_true > 24:
        output = (1,0)
        return output
    elif counter_target_false > 74 and counter_anti_target_false > 74:
        output = (1, 1)
        return output
    else:
        output = (0, 1)
        return output

def allocate(target_allocations, anti_target_allocations, is_target_choices):
    """
    :param target_allocations: A binary list, where True values at index i
            indicate a reward was allocated to the target side on trial i.
    :param anti_target_allocations: A binary list, where True values at index i
            indicate a reward was allocated to the anti-target side on trial i.
    :param is_target_choices: A binary list, where True values at index i
            indicate the target_side was chosen on that index.
            The allocation method
    :return: A two-element tuple where the first binary element indicates
                whether a reward should be allocated to the target side,
                and the second whether a reward should be allocated to the
                anti-target side. To refrain from formatting issues, use the
                output function (see below in this file)
    """
    (target_alternative, anti_target_alternative) = addition(is_target_choices, target_allocations, anti_target_allocations)
    return constrain(target_allocations, target_alternative), \
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
    if allocated_rewards >= TOTAL_REWARDS:
        return 0

    # If there are as many trials left as rewards left, in all remaining trials
    # rewards should be allocated
    current_trial_number = len(previous_allocation)
    remaining_trials = NUMBER_OF_TRIALS - current_trial_number
    remaining_rewards = allocated_rewards
    if remaining_trials == (TOTAL_REWARDS - allocated_rewards):
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
    as_python_elements = [ast.literal_eval(el) for el in as_python_lst]
    return as_python_elements

###############################################################################
# Run
###############################################################################

if __name__ == '__main__':
    # print(REWARDS_BOTH_ALTERNATIVES)
    input = parse_input()
    allocation = allocate(*input)
    output(*allocation)