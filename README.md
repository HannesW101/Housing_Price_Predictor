The aim of this project is to predict future home sales prices using housing data from residential homes in Ames, Iowa, by applying regression techniques such as Random Forest and Gradient Boosting. The dataset comprises 79 explanatory variables that describe nearly every aspect of the homes. Both the training and testing datasets are included in this repository under the 'Data' folder.

We begin by examining the overall distribution of home sales prices against the frequency of each price bracket from the training dataset, as shown in the histogram below.

![Figure_1](https://github.com/HannesW101/Housing_Price_Predictor/assets/86373722/a1b4ce6f-761c-4ce9-9476-8136b6589c15)

The data distribution meets our expectations: most homes are moderately priced, with fewer homes at the extremes. The distribution is positively skewed, however, since we plan to use Random Forest and Gradient Boosting for price prediction, there is typically no need to normalize the data.

Data preprocessing is essential. A quick glance at the training data reveals numerous categorical features that need to be converted to numerical values for our models to understand and learn from the data. We achieve this by first separating the numerical and categorical columns. We then use processing pipelines to handle tasks such as filling missing values in numerical columns with median values, scaling numerical columns to unit variance, filling missing values in categorical columns with the most frequent value, and using one-hot encoding to convert categorical variables into a binary matrix.

Training a Random Forest model on the preprocessed data yields the following histogram of predicted home sales prices against the frequency of each price bracket.

 ![Figure_1](https://github.com/HannesW101/Housing_Price_Predictor/assets/86373722/23b119fc-1f6e-4300-a5a1-4a1797c66538)
 
These results are plausible as the distribution maintains the same positive skew observed in the training data, which aligns with our expectations for housing prices. For further analysis, we will use the mean absolute error (MAE) defined below.

![image](https://github.com/HannesW101/Housing_Price_Predictor/assets/86373722/85faa0a7-d69e-425e-bc9b-e0f947029a11)

Since the true sales values in the test data are unknown, we use cross-validation to split our training data into various folds for testing and training. This produces a cross-validation mean absolute error of:

![image](https://github.com/HannesW101/Housing_Price_Predictor/assets/86373722/671a67f3-0999-4b82-9829-3926d53eca97)

This result is quite satisfactory, however, an attempt to use GridSearch for hyperparameter optimization resulted in worse outcomes than the above. The GridSearch code has been commented out as it takes approximately 1 hour and 20 minutes to run while utilizing 100% of CPU power.

Next, we evaluate the Gradient Boosting model's results to see if it outperforms the Random Forest model. Below is the histogram of predicted home sales prices against the frequency of each price bracket obtained using the Gradient Boosting model.

![Figure_1](https://github.com/HannesW101/Housing_Price_Predictor/assets/86373722/5156dbbe-2f3a-4753-805b-f6209af6e44a)

Similar to the Random Forest model, these results are plausible as the distribution retains the positive skew observed in both the training data and Random Forest predictions, consistent with housing prices. Once again, we use the cross-validation mean absolute error for further analysis, as shown below.

![image](https://github.com/HannesW101/Housing_Price_Predictor/assets/86373722/a18e0ba0-7d17-4fa6-aa1f-1b4ef86b58a6)

As we can see, the error is lower than that obtained from the Random Forest method, indicating that Gradient Boosting should be preferred in this case. Similar to the Random Forest model, using GridSearch for hyperparameter optimization resulted in worse outcomes and is therefore commented out in the code.

Finally, we provide the final predictions of home sales prices in the test data, available under the 'Data' folder in this repository.
