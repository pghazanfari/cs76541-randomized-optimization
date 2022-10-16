import math
from sklearn.metrics import confusion_matrix, make_scorer, precision_score

# Custom scoring functions

class Scorer:
    def __init__(self, fn):
        self.__name__ = fn.__name__
        self.fn = make_scorer(fn)
        
    def __repr__(self):
        return self.__name__

    def __str__(self):
        return self.__name__
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

def specificity(y, y_pred, **kwargs):
    cm = confusion_matrix(y, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    return tn / (tn + fp)

def sensitivity(y, y_pred, **kwargs):
    cm = confusion_matrix(y, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    return tp / (tp + fn)

def ppv(y, y_pred, **kwargs):
    cm = confusion_matrix(y, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    return tp / (tp + fp)

def npv(y, y_pred, **kwargs):
    cm = confusion_matrix(y, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    return tn / (tn + fn)

def npv_precision_avg(y, y_pred, **kwargs):
    return 0.25 * precision_score(y, y_pred, **kwargs) + 0.75 * npv(y, y_pred, **kwargs)

class Specificity(Scorer):
    def __init__(self):
        super().__init__(specificity)

class Sensitivity(Scorer):
    def __init__(self):
        super().__init__(sensitivity)
        
class PPV(Scorer):
    def __init__(self):
        super().__init__(ppv)
        
class NPV(Scorer):
    def __init__(self):
        super().__init__(npv)
        
class NpvPrecisionAvg(Scorer):
    def __init__(self):
        super().__init__(npv_precision_avg)
        

