import cv2


class SVMTrainer:
    def __init__(self):
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    def train(self, training_data, labels):
        self.svm.train(training_data, cv2.ml.ROW_SAMPLE, labels)

    def predict(self, sample):
        return self.svm.predict(sample)
