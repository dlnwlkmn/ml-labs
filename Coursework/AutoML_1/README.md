# AutoML Leaderboard

| Best model   | name                                       | model_type    | metric_type   |   metric_value |   train_time |
|:-------------|:-------------------------------------------|:--------------|:--------------|---------------:|-------------:|
|              | [1_Baseline](1_Baseline/README.md)         | Baseline      | logloss       |       0.563692 |         1.32 |
|              | [2_DecisionTree](2_DecisionTree/README.md) | Decision Tree | logloss       |       0.240871 |        13.93 |
|              | [3_Linear](3_Linear/README.md)             | Linear        | logloss       |       0.208661 |         7.52 |
| **the best** | [Ensemble](Ensemble/README.md)             | Ensemble      | logloss       |       0.169179 |         0.71 |

### AutoML Performance
![AutoML Performance](ldb_performance.png)

### AutoML Performance Boxplot
![AutoML Performance Boxplot](ldb_performance_boxplot.png)

### Features Importance
![features importance across models](features_heatmap.png)



### Spearman Correlation of Models
![models spearman correlation](correlation_heatmap.png)

