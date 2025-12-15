import numpy as np
from collections import Counter


def compute_gini_impurity(target_vector):
    if len(target_vector) == 0:
        return 0.0

    unique, counts = np.unique(target_vector, return_counts=True)
    probabilities = counts / len(target_vector)

    gini = 1.0 - np.sum(probabilities ** 2)
    return gini


def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    sorted_feature = feature_vector[sorted_indices]
    sorted_target = target_vector[sorted_indices]

    unique_features = np.unique(sorted_feature)

    if len(unique_features) == 1:
        return np.array([]), np.array([]), None, None

    thresholds = (unique_features[:-1] + unique_features[1:]) / 2.0

    ginis = []
    valid_thresholds = []

    for threshold in thresholds:
        left_mask = feature_vector < threshold
        right_mask = ~left_mask

        left_target = target_vector[left_mask]
        right_target = target_vector[right_mask]

        if len(left_target) == 0 or len(right_target) == 0:
            continue

        left_gini = compute_gini_impurity(left_target)
        right_gini = compute_gini_impurity(right_target)

        n_left = len(left_target)
        n_right = len(right_target)
        n_total = n_left + n_right

        gini = -(n_left / n_total * left_gini + n_right / n_total * right_gini)

        ginis.append(gini)
        valid_thresholds.append(threshold)

    if len(ginis) == 0:
        return np.array([]), np.array([]), None, None

    ginis = np.array(ginis)
    valid_thresholds = np.array(valid_thresholds)

    best_idx = np.argmax(ginis)
    gini_best = ginis[best_idx]
    threshold_best = valid_thresholds[best_idx]

    return valid_thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key == 'feature_types':
                self._feature_types = value
            elif key == 'max_depth':
                self._max_depth = value
            elif key == 'min_samples_split':
                self._min_samples_split = value
            elif key == 'min_samples_leaf':
                self._min_samples_leaf = value
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) <= self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature_idx in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature_idx]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature_idx]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature_idx])
                clicks = Counter(sub_X[sub_y == 1, feature_idx])

                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0

                    if current_click > 0:
                        ratio[key] = current_count / current_click
                    else:
                        ratio[key] = float('inf')

                sorted_categories = sorted(ratio.items(), key=lambda x: x[1])
                categories_map = {cat: idx for idx, (cat, _) in enumerate(sorted_categories)}

                feature_vector = np.array([categories_map.get(x, -1) for x in sub_X[:, feature_idx]])
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)

            if threshold is None:
                continue

            left_mask = feature_vector < threshold
            n_left = np.sum(left_mask)
            n_right = len(sub_y) - n_left

            if self._min_samples_leaf is not None:
                if n_left < self._min_samples_leaf or n_right < self._min_samples_leaf:
                    continue

            if gini_best is None or gini > gini_best:
                feature_best = feature_idx
                gini_best = gini
                split = left_mask

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [cat for cat, idx in categories_map.items() if idx < threshold]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_val = x[feature_idx]

        if self._feature_types[feature_idx] == "real":
            if feature_val < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if feature_val in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_tree(self):
        return self._tree