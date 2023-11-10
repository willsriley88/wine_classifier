import pandas as pd
import numpy as np
from sklearn import *
from matplotlib import pyplot as plt
import seaborn as sns

class WineClassifier:

    data_path = datasets.load_wine()

    def __init__(self, data_frame=None, x=None, y=None):
        if data_frame is None:
            self.data_frame = pd.DataFrame(WineClassifier.data_path["data"],
                                           columns=WineClassifier.data_path["feature_names"])
        if x is None:
            self.x = self.data_frame.copy()
            self.x["target"] = WineClassifier.data_path["target"]
        if y is None:
            self.y = self.x.pop("target")
            self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(
                self.x, self.y, test_size=.2)

    def train_forrest(self, estimators=50):
        forrest = ensemble.RandomForestClassifier(n_estimators=estimators)
        forrest.fit(self.x_train, self.y_train)
        prediction = forrest.predict(self.x_test)
        forrest.score(self.x_test, self.y_test)
        self.get_cross_val(forrest, self.x_test, self.y_test)
        print(model_selection.cross_val_score(forrest, self.x, self.y, scoring="accuracy").mean())
        print(prediction)
        return prediction

    def train_lin_regression(self):
        lin_mod = linear_model.LogisticRegression()
        lin_mod.fit(self.x_train, self.y_train)
        prediction = lin_mod.predict(self.x_test)
        print(lin_mod.score(self.x_test, self.y_test))
        print(model_selection.cross_val_score(lin_mod, self.x, self.y, scoring="accuracy").mean())

    def find_knn_neighbours(self):
        k_range = range(1,20)
        scores = []
        for k in k_range:
            knn = neighbors.KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.x_train, self.y_train)
            scores.append(knn.score(self.x_test, self.y_test))
        plt.figure()
        plt.xlabel("k count")
        plt.ylabel("model accuracy")
        plt.scatter(k_range, scores)
        plt.grid()
        plt.xticks([i for i in range(0, 35, 5)])
        plt.show()

    def knn_train(self, *args):
        self.x = self.x.filter([args[0], args[1]])
        self.x["target"] = WineClassifier.data_path["target"]
        self.y = self.x.pop("target")
        self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(
            self.x, self.y, test_size=.2)
        knn = neighbors.KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.x_train, self.y_train)
        prediction = knn.predict(self.x_test)
        score = knn.score(self.x_test, self.y_test)
        print(model_selection.cross_val_score(knn, self.x, self.y, scoring="accuracy").mean())
        print(metrics.classification_report(self.y_test, prediction))
        wine_prediction = [i for i in prediction]

        try:
            if args[2] == "prediction":
                return wine_prediction
        except Exception as e:
            print()
        finally:
            return wine_prediction

    def train_svm(self):
        svm_mod = svm.SVC()
        dd = svm_mod.fit(self.x_train, self.y_train)
        prediction = svm_mod.predict(self.x_test)
        print(svm_mod.score(self.x_test, self.y_test))
        print(model_selection.cross_val_score(svm_mod, self.x, self.y, scoring="accuracy").mean())

    def get_cross_val(self, function, x, y):
        method_score = model_selection.cross_val_score(function, x, y)
        print(method_score)
        return method_score

    def check_classifiers(self):
        for i in range(1000):
            score_dict = {}
            all_scores = []

            for i in self.data_frame.keys():
                for j in self.data_frame.keys():
                    if j != i:
                        check_x = self.x.filter([str(i), str(j)])
                        x_train, x_test, y_train, y_test = model_selection.train_test_split(
                            check_x, self.y, test_size=.2)
                        knn = neighbors.KNeighborsClassifier(n_neighbors=15)
                        knn.fit(x_train, y_train)
                        prediction = knn.predict(x_test)
                        score = knn.score(x_test, y_test)
                        score_dict[f'{str(self.x[j].name)}{str(self.x[i].name)}'] = score
                        all_scores.append(score)

            score_keys = list(score_dict.keys())
            score_values = list(score_dict.values())
            highest_score = score_values.index(max(all_scores))
            print(f'{score_keys[highest_score]}":"{max(all_scores)}')
            return f'{score_keys[highest_score]}":"{max(all_scores)}'

    def get_training_split(self):
        test_sizes = [round(float(i * .1), 2) for i in range(1, 9)].__reversed__()
        test_sizes = list(test_sizes)
        knn = neighbors.KNeighborsClassifier(n_neighbors=3)
        plt.figure()

        for test_size in test_sizes:
            scores = []
            for i in range(1, 100):
                self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(self.x, self.y, test_size=1 - test_size)
                knn.fit(self.x_train, self.y_train)
                scores.append(knn.score(self.x_test, self.y_test))
            plt.plot(test_size, np.mean(scores), "r+")
        plt.plot()
        plt.xlabel("training % split")
        plt.ylabel("model accuracy")
        plt.show()

    def plot_roc_curve(self, function=None):
        nclasses = 3
        classifier = function
        classifier.fit(self.x_test, self.y_test)
        x = self.x.filter(["alcohol", "flavanoids"])
        self.x["target"] = WineClassifier.data_path["target"]
        self.y = self.x.pop("target")
        self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(
            self.x, self.y, test_size=.2)
        y_score = classifier.predict_proba(self.x_test)
        y_test_bin = preprocessing.label_binarize(self.y_test, classes=[0, 1, 2])
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(nclasses):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = ['cyan', 'magenta', 'purple']
        plt.plot([0,1], [0,1], "r--")
        for i, color in zip(range(nclasses), colors):
            plt.plot(fpr[i], tpr[i], color=color,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))
        ax.set_xlabel("False Positives")
        ax.set_ylabel("True Positives")
        ax.set_title("ROC Curve of Classifications")
        ax.legend(["y=x", "Wine 0", "Wine 1", "Wine 2"])
        plt.show()

    def show_scatter(self, model=neighbors.KNeighborsClassifier(n_neighbors=3)):
        knn = model
        knn.fit(self.x_train, self.y_train)
        predictions = knn.predict(self.x_test)
        plt.scatter(predictions, self.y_test)
        plt.title('Predicted values v Actual values')
        plt.xlabel('Predictions')
        plt.ylabel('Actual values')
        plt.show()

    def get_confusion_matrix(self, prediction):
        cm = metrics.confusion_matrix(self.y_test, prediction)
        plt.figure(figsize=(8, 7))
        sns.heatmap(cm, annot=True)
        plt.title("confusion matrix")
        plt.ylabel("truth")
        plt.xlabel("prediction")
        plt.show()


wine = WineClassifier()
wine.check_classifiers()
wine.find_knn_neighbours()
wine.get_confusion_matrix(wine.knn_train("alcohol", "flavanoids", "prediction"))
wine.get_cross_val(neighbors.KNeighborsClassifier(n_neighbors=3),
                   x=wine.x_test.filter(["alcohol", "flavanoids"]), y=wine.y_test)
wine.plot_roc_curve(neighbors.KNeighborsClassifier(n_neighbors=3))
wine.get_confusion_matrix(wine.train_forrest())


