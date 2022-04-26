from autoPyTorch import AutoNetClassification

# data and metric imports
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import h5py
import numpy as np
# X, y = sklearn.datasets.load_digits(return_X_y=True)
data_path = "/SSD/dataset/dataset_base_bids_6692_speedtest_net_stnext_leaderboard_medianet_test.h5"
dataset = h5py.File(data_path, 'r')
X = np.array(dataset["input"]).reshape(-1,6000)
y = np.array(dataset["label"]).reshape(-1,1)
print(np.shape(X), np.shape(y))
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
print(y_test)
# running Auto-PyTorch
autoPyTorch = AutoNetClassification("tiny_cs",  # config preset
                                    log_level='info',
                                    max_runtime=300,
                                    min_budget=30,
                                    max_budget=90)

autoPyTorch.fit(X_train, y_train, validation_split=0.3)
y_pred = autoPyTorch.predict(X_test)

print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))
print("Accuracy score", sklearn.metrics.balanced_accuracy_score(y_test, y_pred))
print("Accuracy score", sklearn.metrics.precision_recall_fscore_support(y_test, y_pred))

