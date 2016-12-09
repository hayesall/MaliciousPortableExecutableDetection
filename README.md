### Predicting if a Portable Executable is Malicious or Benign:

[Alexander L. Hayes](https://github.iu.edu/hayesall/)

---

##### Table of Contents:

  1. [Overview](#overview)
  2. [Installation](#installation)
  3. [Usage Instructions](#instructions)
  4. [Background](#background)
  5. [Discussion](#troubleshooting)
  6. [Troubleshooting](#troubleshooting)

---

##### Overview:

A ["Portable Executable"](https://en.wikipedia.org/wiki/Portable_Executable) is a file format used for installation on the Windows operating system (32-bit and 64-bit systems), most commonly known for the `.exe` file format.  This program trains a classifier using [scikit-learn](http://scikit-learn.org/stable/), writing pickle files for the classifier and features.  This model can then be used to classify PE files, outputting "malicious" or "clean."

##### Installation:

  * Prerequisites:
    i. numpy, pandas, pickle, scikit, and scipy (`pip install ...`)
    ii. Python 2.7, bash 4
    iii. If you're on IU sharks or burrow, these should already be installed.
    iv. Developed on RHEL 7
  * `git clone https://github.iu.edu/hayesall/malicious-pe.git`

This *should* work on most Linux flavors running bash version 4 or later (check with `bash --version`), code was developed and tested on RHEL 7. Results on macOS have not been tested and therefore are not guarenteed, but theoretically *should* work. Code has not been tested on Windows, but you are welcome to try it out in PowerShell, IDLE, or the development version of bash for Windows (send Alexander an email if you get everything working).

---

##### Instructions

  * `python learnmodel.py [model]`
     > 1. model can be: AdaBoost, DecisionTree, GNB, GradientBoosting, KNN, RandomForest, NONE
     > 2. specifying `NONE` as the model will train all of them before selecting whichever has the highest precision.

  * Manual:
    a. `python checkfile.py exe-dir/[file]`
    b. `for file in exe-dir/*; do python checkfile.py $file; done`

  * Automatic:
    a. `./verify.sh`
    b. If script throws errors during execution, refer to [the troubleshooting FAQ.](#troubleshooting)

---

##### Background

Algorithms:
  1. [Adaptive Boosting](https://en.wikipedia.org/wiki/AdaBoost)
  2. [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree_learning)
  3. [Gaussian Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes)
  4. [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
  5. [K-nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
  6. [Random Forest](https://en.wikipedia.org/wiki/Random_forest)
  7. "ALL" - compare models
    
[Scipy Documentation](http://scikit-learn.org/stable/modules/ensemble.html)
  
Project Background:
    
Malware Detection through classification has been around for some time now. [Adobe Security](https://github.com/adobe-security) put out a [python command-line classifier](https://github.com/adobe-security/Malware-classifier) meant to "quickly and easily determine if a binary file contain[ed] malware, so [professionals] can develop malware detection signatures faster."  This model was developed with the J48, J48 Graft, PART, and Rigor machine-learning algorithms, trained on 100,000 malicious programs and 16,000 clean ones.

[Te-k](https://github.com/Te-k) implemented [his own version](https://github.com/Te-k/malware-classification) and wrote a [beautiful blog post on the topic](https://www.randhome.io/blog/2016/07/16/machine-learning-for-malware-detection/).

[llSourcell](https://github.com/llSourcell) wrote some wrappers to Te-k's code, but [much of the original was still in tact](https://github.com/llSourcell/antivirus_demo).

Where did I come in?
  * Implement k-nearest neighbors during training.
  * Allow the user to specify a specific algorithm to train with (more on that later)
  * Completely rewrote learning.py (mostly unchanged since Te-k's implementation), changing it to `learnmodel.py`. The original code had almost no documentation, no input checking, and the entire extent of training was done during learning.
  * The old version of testing could test a file and tell whether it was malicious or not, but there wasn't a way for actually testing an array of files with an assortment of algorithms. I pulled examples of common Windows programs and wrote a shell script for testing, validation can now be done for each algorithm in practice rather than simply in theory. (For example: RandomForest tends to perform better based on the 70%/30% train/test split based on the accuracy measurements during cross-validation, but in practice K-nearest neighbors seems to perform better)

---

##### Discussion

I had a lot of hope for k-nearest neighbors, but RandomForest consistently seemed to outperform it during the cross validation step of training. But the difference between the two was slight, typically within a few percentage points from each other across a variety of features.

---

##### Troubleshooting

