# Titanic - Machine Learning from Disaster

This project was developed to address the challenge proposed by the Kaggle competition ['Titanic - Machine Learning from Disaster'](https://www.kaggle.com/competitions/titanic). It provided an opportunity to apply and test knowledge in AI and machine learning.

## Objective

The primary task is to predict, based on passenger characteristics such as social class, ticket fare, and gender, whether a passenger would survive or not. 

A **binary logistic regression model** was employed, with the target variable being 'Survived' (1 for survivors, 0 for those who perished).

## Results

The model produced the following performance metrics:

- **Sensitivity (Recall)**: 0.659722  
- **Specificity**: 0.917453  
- **Accuracy**: 0.813202

These results were obtained using a cutoff threshold of 0.6. Additionally, a confusion matrix was generated to provide further insight into the performance of the model.

![image](https://github.com/user-attachments/assets/b191c90b-b7a2-4f6b-99b6-955b014d33e6)

To further validate the model, an ROC curve was generated, resulting in an **area under the curve (AUC)** of **0.85**, indicating good predictive performance.

![image](https://github.com/user-attachments/assets/39e4265f-f489-4f45-9a48-3b769cf16d22)


Based on these metrics, the model performs well, meeting expectations and fulfilling the objective of predicting passenger survival with reasonable accuracy.
