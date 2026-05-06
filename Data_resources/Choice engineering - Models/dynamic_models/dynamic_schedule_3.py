import sys
import ast

###############################################################################
# Template - Insert your code here
###############################################################################
TOTAL_REWARDS = 25
NUMBER_OF_TRIALS = 100
REWARD = 1
NO_REWARD = 0


def allocate(target_allocations, anti_target_allocations, is_target_choices):
    trial = len(target_allocations)
    # In first four trials put rewards in the target side only
    if trial < 5:
        target_alternative, anti_target_alternative = REWARD, NO_REWARD
        return constrain(target_allocations, target_alternative), \
               constrain(anti_target_allocations, anti_target_alternative)

    # in the 5th trial, neither option is rewarded
    if trial == 5:
        target_alternative, anti_target_alternative = NO_REWARD, NO_REWARD
        return constrain(target_allocations, target_alternative), \
               constrain(anti_target_allocations, anti_target_alternative)


    target_alternative, anti_target_alternative = None, None

    current_delay = 2

    # if agent shows she learnt that every 2nd consecutive choice is rewrded, increase delay
    # agent is not allowed mistakes
    # also, reward the anti-target alternative, assuming agent continues choice of the target
    if sum(is_target_choices[-5:]) == 5 and sum(target_allocations[-5:]) == 2:
        current_delay = 3
        anti_target_alternative = REWARD

    # delay is also incerased to 3 if 2-delay wastes too many resources
    if sum(target_allocations) >= 12:
        current_delay = 3

        # if agent shows she learnt that every 3rd consecutive choice is rewarded, increase delay
    # agent is allowed one mistake (once the counter of consecutive trials is allowed to reset)
    if sum(is_target_choices[-9:]) >= 8 and sum(target_allocations[-9:]) == 2:
        current_delay = 4
        anti_target_alternative = REWARD

    # delay is also incerased to 4 if 3-delay wastes too many resources
    if sum(target_allocations) >= 20:
        current_delay = 4

        # if agent shows she learnt that every 4th consecutive choice is rewarded, increase delay
    # agent is allowed 2 mistakes (twice counter of consecutive trials is reset)
    if sum(is_target_choices[-13:]) >= 11 and sum(target_allocations[-13:]) == 2:
        current_delay = 5
        anti_target_alternative = REWARD

    # if previous choice was of the target
    if is_target_choices and is_target_choices[-1]:
        # if the current number of consecutive choices to get reward is 2
        if current_delay == 2:
            # if the previous choice did not yield reward
            if target_allocations[-1] == NO_REWARD:
                target_alternative = REWARD
            else:
                target_alternative = NO_REWARD

        # if the current number of consecutive choices to get reward is 3
        elif current_delay == 3:
            # if both last 2 choices did not yield reward
            if target_allocations[-2:] == [NO_REWARD, NO_REWARD]:
                target_alternative = REWARD
            else:
                target_alternative = NO_REWARD

        # if the current number of consecutive choices to get reward is 4
        elif current_delay == 4:
            # if all last 3 choices did not yield reward
            if target_allocations[-3:] == [NO_REWARD, NO_REWARD, NO_REWARD]:
                target_alternative = REWARD
            else:
                target_alternative = NO_REWARD

        # if the current number of consecutive choices to get reward is 5
        elif current_delay == 5:
            # if all last 4 choices did not yield reward
            if target_allocations[-4:] == [NO_REWARD, NO_REWARD, NO_REWARD, NO_REWARD]:
                target_alternative = REWARD
            else:
                target_alternative = NO_REWARD
    else:
        target_alternative = NO_REWARD

    # We assume that in the first 40 trials, the user is somewhat
    # likely to sample the anti-target side so we start assigning
    # rewards to that side only after the 40th trial.
    # then, because some users are likely to follow patterns, we reward
    # according to a deterministic pattern every 4 trials assuming that
    # those who understand the pattern will switch to the anti-target
    # only once every 4 trials. In the last 19 trials, the pattern
    # becomes shorter over time
    anti_target_alternative = NO_REWARD
    if (((trial - 41) % 4) == 0) and (40 < trial <= 81):
        anti_target_alternative = REWARD
    elif (trial == 84) or (trial == 87) or (trial == 90) or (trial == 92) or (trial >= 94):
        anti_target_alternative = REWARD

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
REWARDS_BOTH_ALTERNATIVES = '(1, 1)'
REWARD_TARGET_ONLY = '(1, 0)'
REWARD_ANTI_TARGET_ONLY = '(0, 1)'
NO_REWARDS_BOTH_ALTERNATIVES = '(0, 0)'


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
    target_allocations = ast.literal_eval(sys.argv[1])
    anti_target_allocations = ast.literal_eval(sys.argv[2])
    is_target_choices = ast.literal_eval(sys.argv[3])
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


if __name__ == '__main__':
    output(*allocate(*parse_input()))
