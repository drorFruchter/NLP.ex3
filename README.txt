dror012, bober.itay
316083898, 315007724

ex3 - MLE vs HMM models:
Files Included:
main.py - running all tasks
BaseTagger.py - base class for tagger models
MLETagger.py - class for MLE model that implements BaseTagger
BIgramHMMTagger.py - class for Bigram HMM model that implements BsaeTagger
PseudoCreate.py - creating the pseudo map for task e
README


Logged Results:
Running MLE Tagger...
MLE Tagger Error Rates:
Known: 0.0611, Unknown: 0.7482, Total: 0.1199

Running Bigram HMM Tagger...
Bigram HMM Tagger Error Rates:
Known: 0.1394, Unknown: 0.6418, Total: 0.1823

MLE and HMM comparison:
Known Words Improvement: -0.0782
Unknown Words Improvement: 0.1064
Total Words Improvement: -0.0625

Running Bigram HMM Tagger With Smoothing...
Bigram HMM Tagger With Smoothing Error Rates:
Known: 0.1269, Unknown: 0.6442, Total: 0.1711

MLE and Smoothing HMM comparison:
Known Words Improvement: -0.0658
Unknown Words Improvement: 0.1040
Total Words Improvement: -0.0512

HMM and Smoothing HMM comparison:
Known Words Improvement: 0.0125
Unknown Words Improvement: -0.0024
Total Words Improvement: 0.0112

Running Bigram HMM Tagger With Pseudo Words...
Bigram HMM Tagger With Pseudo Words Error Rates:
Known: 0.1534, Unknown: 0.4876, Total: 0.1819

MLE and Pesudo HMM comparison:
Known Words Improvement: -0.0923
Unknown Words Improvement: 0.2607
Total Words Improvement: -0.0621

HMM and Pesudo HMM comparison:
Known Words Improvement: -0.0140
Unknown Words Improvement: 0.1543
Total Words Improvement: 0.0004

Smoothing HMM and Pesudo HMM comparison:
Known Words Improvement: -0.0265
Unknown Words Improvement: 0.1566
Total Words Improvement: -0.0108

Running Bigram HMM Tagger With Pseudo Words + Smoothing...
Bigram HMM Tagger With Pseudo Words + Smoothing Error Rates:
Known: 0.1324, Unknown: 0.4959, Total: 0.1634

MLE and Pseudo + Smoothing HMM comparison:
Known Words Improvement: -0.0713
Unknown Words Improvement: 0.2524
Total Words Improvement: -0.0436

HMM and Pseudo + Smoothing HMM comparison:
Known Words Improvement: 0.0070
Unknown Words Improvement: 0.1460
Total Words Improvement: 0.0189

Smoothing HMM and Pseudo + Smoothing comparison:
Known Words Improvement: -0.0055
Unknown Words Improvement: 0.1484
Total Words Improvement: 0.0077

Pseudo HMM and Pseudo + Smoothing HMM comparison:
Known Words Improvement: 0.0210
Unknown Words Improvement: -0.0083
Total Words Improvement: 0.0185

Top 10 Errors from Confusion Matrix Are:
(133, {'true_tag': 'NNS', 'predicted_tag': 'NN'})
(114, {'true_tag': 'NN', 'predicted_tag': 'NP'})
(91, {'true_tag': 'NP', 'predicted_tag': 'NN'})
(86, {'true_tag': 'JJ', 'predicted_tag': 'NN'})
(45, {'true_tag': 'NNS', 'predicted_tag': 'NP'})
(42, {'true_tag': 'JJ', 'predicted_tag': 'NP'})
(34, {'true_tag': 'VBG', 'predicted_tag': 'NN'})
(26, {'true_tag': 'NP', 'predicted_tag': 'JJ'})
(25, {'true_tag': 'NN', 'predicted_tag': 'JJ'})
(25, {'true_tag': 'JJ', 'predicted_tag': 'VBN'})
