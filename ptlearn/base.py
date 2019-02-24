from inspect import signature
from collections import defaultdict
from pprint import pprint
from .metrics import mae, mse, r2_score


class BaseEstimator(object):
    """Base class for all estimators in scikit-learn
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        if cls.__init__ is object.__init__:
            return list()
        
        sig = signature(cls.__init__)
        return [p for p in sig.parameters.keys() if p != 'self']

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, pprint(self.get_params(deep=False)))


class RegressorMixin(object):

    score_functions = {
        'mae': mae,
        'mse': mse,
        'r2_score': r2_score,
    }

    def score(self, X, y, score_func='r2_score', **kwargs):
        y_pred = self.predict(X)
        if score_func is None:
            score_func = self.score_functions['r2_score']
        elif isinstance(score_func, str):
            score_func = self.score_functions[score_func]
        elif not callable(score_func):
            raise ValueError('score_func is neither available nor callable')
        score_ = score_func(y_true=y, y_pred=y_pred, **kwargs)
        return score_
