---
layout: post
title: Telstra Network Disruptions Competition Writeup
---

The recently finished [Telstra Network Disruptions](https://www.kaggle.com/c/telstra-recruiting-network) recruiting competition attracted 974 participants. I ended up at the 31st spot earning my second Top10% badge. My final score on the private Leaderboard was 0.41577.

The task was to predict the severity of service disruptions on Telstra's network. It was a 3-class classification problem evaluated on multiclass logloss. Here I'll share my approach to solving this problem.

*Code and example notebooks can be found [here](https://github.com/gereleth/kaggle-telstra).*

## Features

The dataset for this competition seemed really simple at first but turned out to have some hidden depths. It was feature engineering that led to getting a good result, rather than ensembling or xgboost tuning. Having read the [solution sharing thread](https://www.kaggle.com/c/telstra-recruiting-network/forums/t/19239/it-s-been-fun-post-your-code-github-links-here-after-the-competition) I see that I wasn't as creative as the top players in my feature engineering efforts. Anyway, here's what I used.

* Locations: label-encoded, location_count feature.
* Events: one-hot encoded common events (10-20 columns), number of events per id. I tried including a few columns of label-encoded events, but that made my results worse.
* Resources: one-hot + number of resources
* Severity types: label-encoded
* Log features:
  * I transformed volume with log(1+x) because the range was very large.
  * I used log(1+volume) columns for common log features (40-60 columns), I also computed various aggregates of volume logarithm by id (count, min, mean, max, std, sum) and used 4 columns of label-encoded log features sorted by volume.
* Target-dependent features:
  * I calculated probabilities of fault severities at every location. These were leave-one-out encoded, meaning that for each training sample I calculated the probability by counting all other training samples except this one and normalizing counts. To deal with zeros I also smoothed the result by adding the overall probabilities with a small weight. This was done inside the cross-validation loop to not use information from out-of-fold samples to avoid overfitting. *This has raised some questions, so I'll point to the relevant code which is [here](https://github.com/gereleth/kaggle-telstra/blob/master/src/telstra_data.py) starting at line 244.*
  * I also tried creating similar aggregates based on events and resources, but that gave me no gains in CV scores.

### The magic feature

As stated in the data description each training and test instance is associated with a location and a time point. Timing information was not explicitly provided but it turned out that it was possible to recover this information from the order of rows in some of the data files. This was referred to on the forums as "the magic feature" as it provided a great boost to the classifier performance.

I discovered it when I wanted to look at some relationships between features and decided to start at locations and severity_types. But when I joined locations to severity types table I noticed that location column was sorted (it went all 1s, all 10s, 100s, ... 999s). So I thought that if rows are sorted by place, maybe they are also sorted by time, and I generated a feature by numbering rows inside each location group. The cv score drop was so sudden that I thought I made some mistake and somehow introduced leakage... But the LB agreed and even allowed me a brief stay in the Top10 =).

Later I also added this variable normalized to (0,1) within each location. Here is an image of normalized count versus location. (green is fault_severity 0, blue is 1, red is 2, test samples are tiny black dots). *See [this notebook](https://github.com/gereleth/kaggle-telstra/blob/master/Discovering%20the%20magic%20feature.ipynb) for how this plot is made.*

![](https://www.dropbox.com/s/58zp2zl6fyctfvx/p7.png?dl=1)

Notice how the lower right part is mostly green. I believe catching that is what brings most of the score improvement.

I also looked at plots of various features versus this supposed 'time'. I took moving averages of logfeature volume aggregates and then used the difference between current value and the moving average as a feature.

I tried a lot to use the moving averages for fault_severities: using only train samples, or also using my own predictions for test set points, but that always resulted in overfitting (nice CV and poor LB scores). I'd love to know if someone made this work.

I also took last known and next known fault severities within a location as features.

## Cross-validation approach

Since the dataset was imbalanced and train set size rather small I used 10-fold cross-validation stratified by fault_severity (target variable).

## Models

### Random Forest

I used RF models (`sklearn.ensemble.RandomForestClassifier`) for experiments evaluating new features, because RF is fast and fairly insensitive to tuning.

Cross-validation scores from RF and similar ET (`sklearn.ensemble.ExtraTreesClassifier`) models of 1000 trees with my final set of features were 0.45-0.46.

### Neural Networks

This was my first time seriously using a neural net model. I adapted [@entron's code](https://www.kaggle.com/c/rossmann-store-sales/forums/t/17974/code-sharing-3rd-place-category-embedding-with-deep-neural-network/103140#post103140) for neural networks with category embedding from Rossmann competition.

The models were built with the keras library. I used the same model structure as @entron only modifying the input and output layers to suit this competition's problem. In between them were 2 dense layers with dropout.

The models used embedding layers for locations and severity types, n-of-k encoded events and resources, columns of log feature volumes and other features described above.

In the first attempts to train the network I noticed heavy overfitting so I proceeded to add regularization coefficients wherever I could. Then layer sizes, dropouts and regularisation coefficients were tuned by hyperopt.

Training time for a single model on my GPU was about 1 minute.

Best configurations I found had single model performance in the range 0.460-0.465, but repeated runs of the same model with different random initializations blended quite nicely and an average of 10 runs was good for score of 0.445. A weighted average of 3 different configurations each having 10 runs gave me a score of 0.440.

*Source code for my NN model is [here](https://github.com/gereleth/kaggle-telstra/blob/master/src/NNmodel.py) and parameter values of the three configurations used can be found [in this notebook](https://github.com/gereleth/kaggle-telstra/blob/master/NN%20and%20XGB%20models%20%2B%20my%20blending%20approach.ipynb).*

### XGBoost models

For xgboost tuning I also used hyperopt.

Parameters I tuned included eta, max\_depth, subsample, colsample\_bytree, lambda and alpha.

I did not tune number of trees, rather I used early stopping in the first fold and the same number of trees for the rest of folds.

I also found that setting max\_delta\_step = 1 as suggested in the docs for imbalanced classification problems helped improve my scores even though I only have a vaguest idea of what this parameter means.

Contrary to the NN models I found that blending xgboost predictions from different runs even with different settings did little to improve the score. So while best single models had a cv score of 0.435-0.44, their ensemble was only barely lower than that.

### Refined random forest models

The approach is described in the paper "[Global Refinement of Random Forest](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ren_Global_Refinement_of_2015_CVPR_paper.pdf)".

The idea is to build a random forest, then fit a regularized linear model using all leaves as binary features (leaf1\_tree1, leaf2\_tree1...,leafM-1\_treeN, leafM_treeN). Then a small proportion (10%) of least significant leaves is pruned, and the linear model is rerun on the new set of leaves. This process is repeated until a desired model size or a desired performance metric is reached. The article claims that this method can improve random forest performance while simultaneously reducing model size.

I initially wrote an implementation of this for the [Caterpillar Tube Pricing](https://www.kaggle.com/c/caterpillar-tube-pricing) competition which was a regression problem. That time I succeeded in making my random forest model 20 times smaller in terms of number of leaves while keeping performance the same. Which was kind of a disappointment. Still I decided to give this approach one more try on a classification problem. I modified the code to use a logistic regression instead of a linear regression. For solving the logistic regression I used `sklearn.linear_model.LogisticRegression` with solver `lbfgs` that can handle multinomial loss with L2 regularisation.

This time I actually saw a performance improvement over the original random forest. For refinement I used a forest of 200 trees. Since my code for iterating over leaves and pruning ran pretty slow I couldn't use cross-validation to select the optimal number of prunings. So in each fold I would run the refinement procedure on the training data while watching performance on the out-of-fold samples. Then with the best model I would predict probabilities for the test set. In the end of cross-validation loop I averaged test predictions from ten folds and took that as the model's test predictions. The number of prunings varied fold to fold and was usually in the range 6 - 20.

Refined RF and ET models had cv scores in the range 0.440-0.445, compared to 0.460 with a regular random forest having the same number of trees.

*I've added a notebook with a demonstration of [RF refinement](https://github.com/gereleth/kaggle-telstra/blob/master/Global%20refinement%20of%20random%20forest.ipynb).*

### Postprocessing predictions

I found that Random forest and Extra Trees predictions benefit from probability calibration. Two calibration methods described, for example, in "[Predicting good probabilities with supervised learning](http://www.datascienceassn.org/sites/default/files/Predicting%20good%20probabilities%20with%20supervised%20learning.pdf)" include Platt scaling and Isotonic regression.

I used a variant of Platt scaling which transforms predicted probabilities using a formula 1/(1+exp(-Ax+B)). I calculated As and Bs for each class simultaneously using `scipy.optimize.minimize` to minimize log loss on out-of-fold predictions for the whole train set. Then I transformed test predictions with the found coefficients. This tended to improve CV score by ~0.010 and LB score by ~0.008.

Calibration with isotonic regression led to overfitting (cv scores improved significantly while LB scores suffered).

XGBoost, neural nets and refined random forests were pretty well calibrated and did not require this treatment. Probably because they worked at optimising the multiclass log loss directly.

### Ensembling approach

Since I managed to make neural nets as well as tree-based models work I could definitely gain something from ensembling them. For this purpose I generated out-of-fold predictions using each model in a 10-fold cross-validation loop with a fixed seed. The seed I selected was one that gave me the lowest standard deviation of fold scores (I ran a couple dozens RF experiments to find it). I took the trouble of finding this splitting because it seemed that the dataset had both "hard" and "easy" samples and I wanted roughly the same concentration of both in each fold. With some random seeds the fold scores could range anywhere from 0.39 to 0.52, but with the one I chose the range was 0.44-0.48.

I tried using a logistic regression for stacking predictions, but that gave me worse scores than a simple linear combination of first layer models.

Then I wanted to find the best weights for this linear combination. I wanted coefficients that would be nonnegative and sum to one, but didn't know how to do such constrained optimisation. But then it hit me that this is exactly what the softmax function is for. I ended up using `scipy.optimize.minimize` to minimize multiclass log loss using out-of-fold predictions from each model. Model weights `w` were related to the values `x` being optimized with `w = np.exp(x)/np.sum(np.exp(x))`.

I applied this process to models within each group (nn, xgb, rf, et, rrf, ret) to get six first layer prediction files. And then applied it again on the six files to generate my final prediction. It ended up being `0.40 nn + 0.25 xgb + 0.25 rrf + 0.1 ret`. RF and ET models received very low weights and were effectively rejected.

The local value of mlogloss on this combination was 0.42100, public LB score was 0.42223, so they were pretty consistent.

Then I employed one more trick based on k nearest neighbors classifier. I used predictions from 4 selected models as features and fit a knn classifier with 100 distance-weighted neighbors. On its own the classifier scored around 0.44, but when blended with my final predictions with ratio 1/3 - 2/3 it dropped my local score and both LB scores by 0.001. Not much, but every little bit helps.

*See [this notebook](https://github.com/gereleth/kaggle-telstra/blob/master/NN%20and%20XGB%20models%20%2B%20my%20blending%20approach.ipynb) for the code of my ensembling approach.*

### Workflow: using Sacred and Hyperopt

In the previous competitions I often found myself overwhelmed by the amount of different models, settings, features I would try out. It was hard to keep track of everything and when I looked for a solution I found Sacred.

[Sacred](https://github.com/IDSIA/sacred) is a tool for organizing, running and logging machine learning experiments. It comes with a mongo observer that saves the run's configuration and results to a mongo database. I found it very useful especially in conjunction with [Hyperopt](https://github.com/hyperopt/hyperopt) which is a Python library for hyperparameter optimization.

With these two tools I made practically no manual tuning of any models for this competition. I would just define a search space over hyperparameters and an objective function that used a Sacred experiment to calculate CV logloss. Then I set it to work overnight, and the next day all configurations and results are just one database query away.

*I've shared a notebook with [an example of automatic model tuning](https://github.com/gereleth/kaggle-telstra/blob/master/Automatic%20model%20tuning%20with%20Sacred%20and%20Hyperopt.ipynb).*

## Conclusions

I had a lot of fun participating in this competition. The small dataset size allowed for quick experimentation and trying a lot of things. I got some experience using the neural net models. I added Sacred and Hyperopt to my toolbox.
I combed the solution sharing thread for feature engineering ideas that never even occured to me before (hopefully they will next time). I even wrote and shared a lengthy report =).

Thanks to Kaggle and Telstra for organizing this competition, congratulations to the winners and cheers to all participants!
