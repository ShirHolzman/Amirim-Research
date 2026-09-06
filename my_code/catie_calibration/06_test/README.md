# 06_test

Purpose: score the frozen ladder on the Test split (schedules 1/8/10) exactly once, after everything else is final.
Input: Test cache (built with an explicit flag; absent until then). Output: test.csv. No prints.
Script: test_split.py (to be written). Golden check first: M0 on Test must reproduce 01_bug_correction's Test row before anything else is scored.
