from sklearn.base import RegressorMixin, BaseEstimator, is_regressor


class DistilledRegressor(BaseEstimator, RegressorMixin):
    """
    Class to implement distillation. Currently only supports regression.
    Params
    ------
    teacher: initial model to be trained
    student: model to be distilled from teacher's predictions
    """

    def __init__(self, teacher: BaseEstimator, student: BaseEstimator):
        self.teacher = teacher
        self.student = student
        self._validate_student()
        self._check_teacher_type()

    def _validate_student(self):
        if is_regressor(self.student):
            pass
        else:
            if not hasattr(self.student, "prediction_task"):
                raise ValueError("Student must be either a scikit-learn or imodels regressor")
            elif self.student.prediction_task == "classification":
                raise ValueError("Student must be a regressor")

    def _check_teacher_type(self):
        if hasattr(self.teacher, "prediction_task"):
            self.teacher_type = self.teacher.prediction_task
        elif hasattr(self.teacher, "_estimator_type"):
            if is_regressor(self.teacher):
                self.teacher_type = "regression"
            else:
                self.teacher_type = "classification"

    def set_teacher_params(self, **params):
        self.teacher.set_params(**params)

    def set_student_params(self, **params):
        self.student.set_params(**params)

    def fit(self, X, y, **kwargs):
        self.teacher.fit(X, y, **kwargs)
        if self.teacher_type == "regression":
            preds = self.teacher.predict(X)
        else:
            preds = self.teacher.predict_proba(X)[:, 1]
        self.student.fit(X, preds)

    def predict(self, X):
        return self.student.predict(X)
