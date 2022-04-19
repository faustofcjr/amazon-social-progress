# Project Checklist

1. Frame the problem and look at the big picture
2. Get the data
3. Explore the data
4. Prepare the data
5. Short-list promising models
6. Fine-Tune the System 
7. Present your solution
8. Launch!
9. Best Kaggle Entry!

## 1. Frame the problem and look at the big picture

### 1.1 Define the objective in business terms
### 1.2 What are the current solutions/workarounds (if any)?

### 1.3 How should you frame this problem?
- [ ] Supervised learning
- [ ] Unsupervised learning
- [ ] Reinforcement learning
- [x] Online
- [ ] Offline
### 1.4 How should performance be measured?
### 1.5 Is the performance measure aligned with the business objective?
- [ ] Yes
- [ ] No
### 1.6 What would be the minimum performance needed to reach the business objective?
### 1.7 What are comparable problems? Can you reuse experience or tools?
### 1.8 Is human expertise available?
- [x] Yes
- [ ] No
### 1.9. How would you solve the problem manually?
### 1.10 List the assumptions you or others have made so far
### 1.11 Verify assumptions if possible

## 2. Get the data

### 2.1 List the data you need and how much you need
### 2.2 Find and document where you can get that data
### 2.3 Check how much space it will take
### 2.4 Check legal obligations, and get the authorization if necessary
### 2.5 Get access authorizations
### 2.6 Create a workspace (with enough storage space)
- [ ] Yes
- [ ] No

### 2.7 Get the data
- [x] Done
- [ ] Not yet
### 2.8 Convert the data to a format you can easily manipulate (without changing the data itself)
- [x] Data properly formatted
- [ ] Data were splitted in train, validation and test
- [ ] Data need a lot of tweaking
### 2.9 Ensure sensitive information is deleted or protected (e.g., anonymized)
- [x] Done
- [ ] Not yet
### 2.10 Check the size and type of data (time series, sample, geographical, etc.)
- [ ] Yes
- [ ] No
### 2.11 Sample a test set, put it aside, and never look at it (no data snooping!)
- [ ] Yes
- [ ] No

## 3. Explore the data
<!--- <details>
<summary>Check</summary> and here
-->

### 3.1 Create a copy of the data for exploration (sampling it down to a manageable size if necessary).
- [x] Done
- [ ] Not yet

### 3.2 Create a Jupyter notebook to keep record of your data exploration.
- [x] Done
- [ ] Not yet

### 3.3 Study each attribute and its characteristics:
- [x] Names
- [x] Type (categorical, int/float, bounded/unbounded, text, structured, etc.)
- [x] % of missing values
- [x] Noisiness and type of noise (stochastic, outliers, rounding errors, etc.)
- [ ] Possibly useful for the task?
- [ ] Type of distribution (Gaussian, uniform, logarithmic, etc.)

### 3.4 For supervised learning tasks, identify the target attribute(s).
### 3.5 Visualize the data.
### 3.6 Study the correlations between attributes.
- [ ] Yes
- [ ] No
### 3.7 Study how you would solve the problem manually.
- [ ] Yes
- [ ] No
### 3.8 Identify the promising transformations you may want to apply.
### 3.9 Identify extra data that would be useful.
- [ ] Yes
- [ ] No
### 3.10 Document what you have learned.
- [ ] Yes
- [ ] No
<!--- </details> -->

## 4. Prepare the data

### 4.1 Keep the original dataset intact, Work on copies of it.
- [x] Done
- [ ] Not yet

### 4.2 Write functions for all data transformations you apply.
```
For five reasons:
1º So you can easily prepare the data the next time you get a fresh dataset  
2º So you can apply these transformations in future projects  
3º To clean and prepare the test set  
4º To clean and prepare new data instances  
5º To make it easy to treat your preparation choices as hyperparameters
```  
- [ ] Yes
- [ ] No

### 4.3 Data cleaning:  
- [x] Fix or remove outliers (optional).  
- [ ] Fill in missing values (e.g., with zero, mean, median...) or drop their rows (or columns).

### 4.4 Feature selection (optional):  
- [ ] Drop the attributes that provide no useful information for the task.  

### 4.5 Feature engineering, where appropriates:  
- [ ] Discretize continuous features.  
- [ ] Decompose features (e.g., categorical, date/time, etc.).  
- [ ] Add promising transformations of features (e.g., log(x), sqrt(x), x^2, etc.).
- [ ] Aggregate features into promising new features.  

### 4.6 Feature scaling: standardize or normalize features.

## 5. Short-list promising models
```
Notes: 

- If the data is huge, you may want to sample smaller training sets, so you can train many different models in a reasonable time (be aware that this penalizes complex models such as large neural nets or Random Forests). 

- Once again, try to automate these steps as much as possible.    

```

### 5.1 Train many quick and dirty models from different categories (e.g., linear, naive, Bayes, SVM, Random Forests, neural net, etc.) using standard parameters.
- [ ] Decision Tree Classifier
- [ ] K-Nearest Neighbors
- [ ] Ensemble Random Forests
- [ ] Ensemble Adaboost

### 5.2. Measure and compare their performance.  
```
For each model, use N-fold cross-validation and compute the mean and standard deviation of their performance. 
```
3. Analyze the most significant variables for each algorithm.  
4. Analyze the types of errors the models make.  
    - What data would a human have used to avoid these errors?  
5. Have a quick round of feature selection and engineering.  
6. Have one or two more quick iterations of the five previous steps.  
7. Short-list the top three to five most promising models, preferring models that make different types of errors.

## 6. Fine-Tune the System  
Notes:  
- You will want to use as much data as possible for this step, especially as you move toward the end of fine-tuning.   
- As always automate what you can.    

1. Fine-tune the hyperparameters using cross-validation.  
    - Treat your data transformation choices as hyperparameters, especially when you are not sure about them (e.g., should I replace missing values with zero or the median value? Or just drop the rows?).  
    - Unless there are very few hyperparamter values to explore, prefer random search over grid search. If training is very long, you may prefer a Bayesian optimization approach (e.g., using a Gaussian process priors, as described by Jasper Snoek, Hugo Larochelle, and Ryan Adams ([https://goo.gl/PEFfGr](https://goo.gl/PEFfGr)))  
2. Try Ensemble methods. Combining your best models will often perform better than running them invdividually.  
3. Once you are confident about your final model, measure its performance on the test set to estimate the generalization error.

> Don't tweak your model after measuring the generalization error: you would just start overfitting the test set.  

## 7. Present your solution
1. Document what you have done.  
2. Create a nice presentation.  
    - Make sure you highlight the big picture first.  
3. Explain why your solution achieves the business objective.  
4. Don't forget to present interesting points you noticed along the way.  
    - Describe what worked and what did not.  
    - List your assumptions and your system's limitations.  
5. Ensure your key findings are communicated through beautiful visualizations or easy-to-remember statements (e.g., "the median income is the number-one predictor of housing prices").

## 8. Launch!
1. Get your solution ready for production (plug into production data inputs, write unit tests, etc.).  
2. Write monitoring code to check your system's live performance at regular intervals and trigger alerts when it drops.  
    - Beware of slow degradation too: models tend to "rot" as data evolves.   
    - Measuring performance may require a human pipeline (e.g., via a crowdsourcing service).  
    - Also monitor your inputs' quality (e.g., a malfunctioning sensor sending random values, or another team's output becoming stale). This is  particulary important for online learning systems.  
3. Retrain your models on a regular basis on fresh data (automate as much as possible).
