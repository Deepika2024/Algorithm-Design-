# Fake Account Detection using Graph Convolutional Network (GCN)

Algorithm Design assignment. A GCN implemented in pure Python
(no NumPy, no ML libraries) to detect fake social-media accounts.

## Idea
Accounts are nodes, follows/interactions are edges. Fake accounts tend to
connect with other fake accounts. With only 4 verified accounts, the GCN
predicts whether the other accounts are real or fake using both account
features and the connections.

## Formula
Output = softmax( Â · ReLU(Â · X · W1) · W2 ),  Â = D^(-1/2)(A + I)D^(-1/2)

## Run
```bash
python fake_account_gcn.py
```

## Result
All 10 unlabeled accounts classified correctly (see output.txt).

## Complexity
Dense: time O(T·n²·(f+h)), space O(n²).
Sparse: time O(T·(e·(f+h) + n·f·h)), space O(n+e).

## Limitations
Small synthetic dataset; fake accounts that connect only to real users can
evade graph-based detection; very deep GCNs over-smooth.

## Reference
Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional
Networks", ICLR 2017.
