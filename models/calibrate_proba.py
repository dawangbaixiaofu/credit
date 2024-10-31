"""
reference: 
https://scikit-learn.org/1.5/api/sklearn.calibration.html


class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    calibrated_classifiers_: list = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    classes_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        estimator: None | BaseEstimator = None,
        *,
        method: Literal["sigmoid", "isotonic", "sigmoid"] = "sigmoid",
        cv: int | BaseCrossValidator | Iterable | None | str | BaseShuffleSplit = None,
        n_jobs: None | Int = None,
        ensemble: bool = True,
        base_estimator: str | BaseEstimator = "deprecated",
    ) -> None: ...
    def fit(
        self: CalibratedClassifierCV_Self,
        X: MatrixLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
        **fit_params,
    ) -> CalibratedClassifierCV_Self: ...
    def predict_proba(self, X: MatrixLike) -> ndarray: ...
    def predict(self, X: MatrixLike) -> ndarray: ...

-----------------------------------------------------------------------------------------------




"""

from sklearn.calibration import CalibratedClassifierCV



from sklearn.isotonic import IsotonicRegression
from sklearn.isotonic import check_increasing
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import numpy as np 
import pandas as pd 


class Calibrator:
    '''
    sklearn.calibrate.CalibratedClassifierCV
    实现的概率校验是使用(pred_proba, y)进行拟合，而非在每个分箱内的真实标签概率值。

    Parameters
    ----------
    X : estimator predict probability 
    y : true label = 1 sample ratio in each bin which is made by prob, 
        or true class
    '''
    def __init__(self, method="isotonic") -> None:
        self.method = method
        

    def fit(self, X, y, **fit_params):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.method == "isotonic":
            self.clf = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip')
        elif self.method == "sigmoid":
            self.clf = LogisticRegression()
            X = X.reshape(-1, 1)
        
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if self.method == "isotonic":
            proba = self.clf.predict(X)
        elif self.method == "sigmoid":
            X = X.reshape(-1, 1)
            proba = self.clf.predict_proba(X)[:, 1]
        return proba


    

def gen_calibration_curve_data(y_prob, y_true, n_bins=10):
    '''
    * 等距划分
    分位数划分
    等频划分
    '''
    bins:list = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0 # 箱内样本数量为空即分母为0的去除
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return prob_pred, prob_true




def make_calibrate_data(n_samples):
    # 使用分类模型产生概率，而非直接生成概率
    X, y = make_classification(n_samples=n_samples, n_features=20, n_informative=10, n_redundant=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = LogisticRegression().fit(X_train, y_train)
    pred_proba = estimator.predict_proba(X_test)[:, 1]


    # pred_proba = np.random.uniform(size=n_samples)
    # _max = np.max(pred_proba)
    # _min = np.min(pred_proba)
    # pred_proba = (pred_proba-_min)/(_max - _min)

    # label = []
    # error = 0.2
    # for p in pred_proba:
    #     choice_proba = p + error * np.random.randint(low=-2, high=3)
    #     if choice_proba > 1:
    #         choice_proba = 1
    #     elif choice_proba < 0:
    #         choice_proba = 0
    #     random_label = np.random.choice(a=[0,1], p=[1-choice_proba, choice_proba])
    #     label.append(int(random_label))

    return pred_proba, y_test


"""
figure:
perfect curve VS calibrated curve VS uncalibrated curve
"""
import matplotlib.pyplot as plt 

def calibrate_curve(proba_predict, proba_calibration, proba_true_uncalibrate, proba_true_calibrate):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(proba_predict, proba_true_uncalibrate, label="uncalirate", linestyle="-.", color='red')
    ax.plot(proba_calibration, proba_true_calibrate, label="calibrate", linestyle="--", color='green')
    # perfect curve
    perfect_x = [x/100 for x in range(101)]
    perfect_y = perfect_x
    ax.plot(perfect_x, perfect_y, label="perfect", linestyle="-", color='gray')

    ax.legend()
    ax.set_title("Calibration Probability Figure")
    ax.set_xlabel("Probability Prediction")
    ax.set_ylabel("Probability True")
    plt.show()

def isotonic_curve(proba_predict, proba_calibrate):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(proba_predict, proba_calibrate, label="isotonic", linestyle="-")
    ax.set_title("Isotonic Regression Figure")
    ax.set_xlabel("Probability Prediction")
    ax.set_ylabel("Probability Isonotic")
    plt.show()


if __name__ == "__main__":
    # 测试集
    proba_predict, y = make_calibrate_data(n_samples=5000)
    calibrator = Calibrator(method="isotonic")
    calibrator.fit(proba_predict, y)
    proba_calibration = calibrator.predict_proba(proba_predict)
    df = pd.DataFrame({'proba_predict':proba_predict, 'y':y})
    df['proba_calibration'] = proba_calibration
    print(df.head())
    

    # data saved for test
    df.to_csv(path_or_buf="./test_calibration_data.csv", index=False)
    
    # visual
    proba_pred_bin, proba_true_bin = gen_calibration_curve_data(proba_predict, y)
    proba_calibration_bin, proba_calibration_true_bin = gen_calibration_curve_data(proba_calibration, y)
    calibrate_curve(proba_pred_bin, proba_calibration_bin, proba_true_bin, proba_calibration_true_bin)
    
    # visual isotonic regression figure
    isotonic_data = [(x,y) for x, y in zip(proba_predict, proba_calibration)]
    isotonic_data_sorted = sorted(isotonic_data, key=lambda x:x[0])
    proba_predict_sorted = [x[0] for x in isotonic_data_sorted]
    proba_calibrate_sorted = [x[1] for x in isotonic_data_sorted]
    isotonic_curve(proba_predict_sorted, proba_calibrate_sorted)

    # check bined data
    data_uncalibrate = {'proba_pred_bin':proba_pred_bin, 
                        'proba_true_bin':proba_true_bin
                        }
    data_calibrate = {
        'proba_calibration_bin':proba_calibration_bin, 
        'proba_calibration_true_bin':proba_calibration_true_bin
    }
    df_uncalibrate = pd.DataFrame(data_uncalibrate)
    df_calibrate = pd.DataFrame(data_calibrate)
    df_uncalibrate.to_csv(path_or_buf='./uncalibrate_curve_data.csv', index=False)
    df_calibrate.to_csv(path_or_buf='./calibrate_curve_data.csv', index=False)