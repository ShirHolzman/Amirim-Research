# This file provides an explanation and example usage of a dynamic allocation
# file for the bias induction competition. For more details on the competition,
#  see [here](http://decision-making-lab.com/competition/index.html)

###############################################################################
# What is a dynamic allocation model?
###############################################################################
# A dynamic allocation model is used to determine the rewards in a single trial
#  of an experiment run as part of the bias induction competition. This output
#  should indicate the allocation of rewards for the two alternatives used in
#  the experiment in a specific trial. The rewards available for allocation are
#  binary (either 1 or 0) and are constrained such that during the 100 trials
#  of the experiment, each of the alternatives should be allocated exactly 25
#  rewards (i.e. 1's, and 75 should be allocated with 0's). The input of the
#  dynamic allocation model is current experiment's history, namely previous
#  allocations and their respective choices.
#
# The goal of the competition, and the dynamic allocation model, is to design
# an allocation mechanism that would maximize the choices in one specific
# alternative, termed the "target alternative". In the experiment, the target
# alternative will be placed randomly either on the left or the right part of
# the participant's screen. However, in the output of the allocation model, the
#  target alternative should be always placed first.

###############################################################################
# How should a dynamic allocation model file be used
###############################################################################
# In the course of an experiment, your dynamic allocation model would be called
#  repeatedly, once for every trial, and the allocation provided by the
#  will be revealed to the participant, according to her choice.
#
# A. Receiving input: your model will be called with two command-line arguments
#  that you may parse
# e.g. with [sys.argv](https://docs.python.org/3.7/library/sys.html#sys.argv)).
#     1. The first input is a list of previous allocations to the target
#      alternative. Each entry in the list is either 1, indicating that in that
#      index a reward was allocated, or 0, indicating that it was not. For
#      example, the list [1, 0, 0, 0] received as the first input for your
#      model indicates that the experiment is currently at its 5'th trial,
#      that on the first trial a reward was allocated to the target alternative
#      and that no rewards were allocated in trial 2, 3, and 4.
#     2. The second input is in the same format as the first, but indicating
#      previous rewards to the second ("anti-target") alternative. Hence, for
#      example, the list [0, 1, 1] as the second input to your model indicates
#      that it is currently the 4'th trial, that on the first trial rewards
#      were not allocated to the anti-target alternative and that on the second
#      and third trial rewards were allocated to that alternative.
#     3. The third input is a list of previous choices, where 1's indicate a
#      choice in the target alternative and 0's indicate choice in the
#      anti-target alternative. For example, the list [1, 1, 1, 0, 0, 0]
#      received as the third input indicates that it is currently the 7'th
#      trial, that on the first three trials the target side was chosen by the
#      user and that on the last three trials it was not.
# B. Providing output - Your model should indicate the allocation of rewards
# by printing (to the standard sys.stdout, e.g. using print) a single string in
#  the format of "(T, N)", where both T and N are either the character 1 or 0,
#  T represents the allocation to the target side and N the allocation the
#  non-target side.
# Hence, your model output should be one of the following four strings
# "(0, 0)",  "(0, 1)",  "(1, 0)",  "(1, 1)". To prevent formatting issues,
# you may use the "output" function provided in this file.

###############################################################################
# Code start
###############################################################################
import sys
import ast
###############################################################################
# Template - Insert your code here
###############################################################################
TOTAL_REWARDS = 25
NUMBER_OF_TRIALS = 100
REWARD = 1
NO_REWARD = 0
print_deb = False
def debug_print(ttt):
    if print_deb:
        print(ttt)

NO_REWARD = 0
REWARD = 1

def allocate(target_allocations, anti_target_allocations, is_target_choices):
    """
    :param target_allocations: A binary list, where True values at index i
            indicate a reward was allocated to the target side on trial i.
    :param anti_target_allocations: A binary list, where True values at index i
            indicate a reward was allocated to the anti-target side on trial i.
    :param is_target_choices: A binary list, where True values at index i
            indicate the target_side was chosen on that index.
    :return: A two-element tuple where the first binary element indicate
                whether a reward should be allocated to the target side,
                and the second whether a reward should be allocated to the
                anti-target side.
    """
    
        
    # f = open('C:\\xampp\\htdocs\\test','a')
    # f.write("\ntarget_allocations = " + str(target_allocations))
    # f.write("\nanti_target_allocations = " + str(anti_target_allocations))
    # f.write("\nis_target_choices = " + str(is_target_choices))
    
    # f.write("\n")

    # f.close()
    
    
    trial = len(target_allocations)
    real_trial = trial

    debug_print('real trial ' + str(trial))
    restart_bonus = False
    trial_reset = 0
    ## search for 10 without bonus
    if trial>11:
        non_bonus_len = 10
        for k in range(len(is_target_choices)-non_bonus_len+1):
            last_is_target_choices = is_target_choices[k:k+non_bonus_len]
            last_target_allocations = target_allocations[k:k+non_bonus_len]
            got_reward_in_last_ten = 0 
            for i in range(len(last_is_target_choices)):
                if last_is_target_choices[i]==True and last_target_allocations[i]==1:
                    got_reward_in_last_ten =1
            if got_reward_in_last_ten == 0:
                trial_reset = k+10
                restart_bonus= True
    debug_print('trial ' + str(trial))
    if restart_bonus == False:
        d0,d1,d2,d3,d4,d5 = [10,25,40,60,80,90]
    else:
        zt = [[2,0.1],[3,0.15],[4,0.15],[5,0.2],[4,0.2],[3,0.1]]
        num_prize_we_still_have = 25-sum(target_allocations)
        added_b = num_prize_we_still_have - int(sum([round(l[1]*(100-real_trial))/l[0] for l in zt]))
        d0,d1,d2,d3,d4,d5 = [round(l[1]*(100-trial_reset)) for l in zt]
        debug_print(str([d0,d1,d2,d3,d4,d5]))
        #if we have bonus left, we increase the length of the z==2
        if added_b>0:
            d1+=added_b*2

    
    if trial==0:
        return constrain(target_allocations, NO_REWARD), constrain(anti_target_allocations, NO_REWARD)
    z = 2    #So z-1 is the number of null choices needed to trigger the prize  
    t10 =0
    first_3z = 1 # first time we are in z == 3
    if True in is_target_choices and restart_bonus == False:
        first_true = is_target_choices.index(True)
        t10 = int(first_true/10)*10
    debug_print('d0 {} t10 {}'.format(d0,t10))
    debug_print('trial {}'.format(trial))
    debug_print('trial-trial_reset {}'.format(trial-trial_reset))
    
    
    if (trial-trial_reset)>d0+t10:
        z = 3 
        
    if (trial-trial_reset)>d1+t10:
        first_3z = 0
        z = 4 
        
    if (trial-trial_reset)>d2+t10:
        z = 5 
        
    if (trial-trial_reset)>d3+t10:
        z = 4 
        
    if (trial-trial_reset)>d4+t10:
        z = 3 
        
    if (trial-trial_reset)>d5+t10:
        z = 3 
    
    if True not in is_target_choices :
        z=2        
    debug_print(z)
    if 1 in target_allocations:
        debug_print('I am ')
        index_of_last_allocation = len(target_allocations) - target_allocations[-1::-1].index(1) - 1
        debug_print(index_of_last_allocation)
        if real_trial-index_of_last_allocation>=z:
            if all(is_target_choices[(-z+1):]):
                if z==2:
                    return constrain(target_allocations, REWARD), constrain(anti_target_allocations, NO_REWARD)
                elif z==3 and first_3z ==1:
                    for ii in range(len(target_allocations)-2):
                        if target_allocations[ii:ii+3]==[0,0,1] and is_target_choices[ii:ii+3] == [True,True,True]:
                            return constrain(target_allocations, REWARD), constrain(anti_target_allocations, REWARD)
                    return constrain(target_allocations, REWARD), constrain(anti_target_allocations, NO_REWARD)
                else:
                    return constrain(target_allocations, REWARD), constrain(anti_target_allocations, REWARD)
    else:
        if all(is_target_choices[(-z+1):]):
            if z==2:
                return constrain(target_allocations, REWARD), constrain(anti_target_allocations, NO_REWARD)
            else:
                return constrain(target_allocations, REWARD), constrain(anti_target_allocations, REWARD)
    return constrain(target_allocations, NO_REWARD), constrain(anti_target_allocations, NO_REWARD)
    
#[1,1,1] , [0,1,0] , [True, False, False]

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
    
    #print("ERROR")
###############################################################################
# Debug
# if you want it to output to
# http://decision-making-lab.com/visual_experiment/cmptn_remote/scripts/display_params.php
# disable any other prints (specifically the one from "output")
###############################################################################
counter = 0
text_agg = ''
def debug(txt="hi"):
    global counter
    global text_agg
    counter = counter + 1
    text_agg = text_agg + " ### " + str(counter) + ":" + str(txt)
    print(text_agg)

def parse_lst(lst):
    as_python_lst = lst.strip('[').strip(']').split(',')
    as_python_elements = [ast.literal_eval(el) for el in as_python_lst]
    return as_python_elements

###############################################################################
# Run
###############################################################################

if __name__ == '__main__':
    input = parse_input()
    allocation = allocate(*input)
    ##constrain here!
    
    output(*allocation)
