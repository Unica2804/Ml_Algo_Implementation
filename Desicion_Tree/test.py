# test_decision_tree.py
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SkTree
from Desicion_Tree import Decisiontree   

# ------------------------------------------------------------------
# 1. 100 % numeric (Iris)
# ------------------------------------------------------------------
print("=== 1. Iris (numeric only) ===")
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

perm = np.random.RandomState(0).permutation(len(y))
X, y = X[perm], y[perm]
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

my_tree = Decisiontree(max_depth=3, min_samples_split=2)
my_tree.fit(X_train, y_train)
my_pred = my_tree.predict(X_test)
print("My accuracy :", accuracy_score(y_test, my_pred))

sk_tree = SkTree(criterion='gini', max_depth=3, min_samples_split=2, random_state=0)
sk_tree.fit(X_train, y_train)
print("Sk accuracy :", accuracy_score(y_test, sk_tree.predict(X_test)))

# ------------------------------------------------------------------
# 2. Mixed numeric / categorical synthetic data
# ------------------------------------------------------------------
print("\n=== 2. Mixed data (numeric + categorical) ===")
rng = np.random.default_rng(42)

# colour, size, weight -> label
n = 500
colour = rng.choice(['red', 'green', 'blue'], n)
size   = rng.choice(['S', 'M', 'L'], n)
weight = rng.normal(0, 1, n)

# simple rule: heavy + blue -> 1, else 0
y_mixed = ((weight > 0.5) & (colour == 'blue')).astype(int)

# build 3-column matrix: colour, size, weight
X_mixed = np.column_stack([colour, size, weight])

# 80-20 split
idx = np.arange(n)
rng.shuffle(idx)
train_idx, test_idx = idx[:400], idx[400:]

X_tr, X_te = X_mixed[train_idx], X_mixed[test_idx]
y_tr, y_te = y_mixed[train_idx], y_mixed[test_idx]

# tell our tree that columns 0 and 1 are categorical
mixed_tree = Decisiontree(categorical_features=[0, 1], max_depth=4, min_samples_split=5)
mixed_tree.fit(X_tr, y_tr)
mixed_pred = mixed_tree.predict(X_te)
print("Mixed accuracy (my tree):", accuracy_score(y_te, mixed_pred))

# sklearn needs one-hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

pre = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), [0, 1]),
        ('num', 'passthrough', [2])
    ])
X_tr_sk = pre.fit_transform(X_tr)
X_te_sk = pre.transform(X_te)

sk_mixed = SkTree(criterion='gini', max_depth=4, min_samples_split=5, random_state=0)
sk_mixed.fit(X_tr_sk, y_tr)
print("Mixed accuracy (sklearn):", accuracy_score(y_te, sk_mixed.predict(X_te_sk)))

# ------------------------------------------------------------------
# 3. Edge-case: pure categorical
# ------------------------------------------------------------------
print("\n=== 3. Pure categorical toy set ===")
X_cat = np.array([['A', 'X'],
                  ['A', 'Y'],
                  ['B', 'X'],
                  ['B', 'Y'],
                  ['C', 'X']], dtype=object)
y_cat = np.array([0, 0, 1, 1, 0])

cat_tree = Decisiontree(categorical_features=[0, 1], max_depth=2)
cat_tree.fit(X_cat, y_cat)
print("Predictions:", cat_tree.predict(X_cat))
print("True labels :", y_cat)
print("Accuracy    :", accuracy_score(y_cat, cat_tree.predict(X_cat)))

print("\nAll tests finished.")