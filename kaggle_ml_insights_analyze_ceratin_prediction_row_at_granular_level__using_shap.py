#Reference: https://www.kaggle.com/dansbecker/shap-values ; https://towardsdatascience.com/one-feature-attribution-method-to-supposedly-rule-them-all-shapley-values-f3e04534983d
#SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value.
#Detailed Explanation: https://towardsdatascience.com/one-feature-attribution-method-to-supposedly-rule-them-all-shapley-values-f3e04534983d

#Plot link: https://i.imgur.com/JVD2U7k.png
#Interpretation : Feature values causing increased predictions are in pink, and their visual size shows the magnitude of the feature's effect. Feature values decreasing the prediction are in blue. 
#SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value.

#Code
import shap  # package used to calculate Shap values
row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired; May need to typecaste into float (if this existed in any other datatype) : data_for_prediction = val_X.iloc[0,:].astype(float)
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model) 
# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

my_model.predict_proba(data_for_prediction_array)

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)



# use Kernel SHAP to explain test set predictions : shap.KernelExplainer - KernelExplainer to get the same results
k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

#shap.DeepExplainer works with Deep Learning models
