# 02_reliability

Purpose: reliability diagrams of CATIE's forecast on the Training split (bin-count sweep, doubled/action-based, folded-by-run-length), plus one EDA replication panel generated last.
Input: cache/ (Training, EDA). Model: corrected CATIE, published parameters.
Output: figures/ and reliability.csv (one row per bin per diagram per split). No prints.
Script: reliability.py (to be written; <=150 lines). Test: tests/02_reliability/reliability_test.py.
