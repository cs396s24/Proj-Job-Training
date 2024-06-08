import pandas as pd
from CTL.causal_tree_learn import CausalTree
from sklearn.model_selection import train_test_split
import numpy as np

lalonde_psid = pd.read_csv('../data/lalonde_exp.csv')
# lalonde_psid['re78'] = lalonde_psid['re78'] - lalonde_psid['re75']
lalonde_psid.drop(['nodegree', 'u75', 'u74'], axis=1, inplace=True)
y = lalonde_psid['re78'].values
treatment = lalonde_psid['treat'].values

x = lalonde_psid.drop(['treat', 're78'], axis=1).values

columns = lalonde_psid.drop(['treat', 're78'], axis=1).columns

x_train, x_test, y_train, y_test, treat_train, treat_test = train_test_split(x, y, treatment,
                                                                             test_size=0.5, random_state=42)

# regular CTL
ctl = CausalTree(magnitude=False)
ctl.fit(x_train, y_train, treat_train)
ctl.prune()
ctl_predict = ctl.predict(x_test)

ctl.plot_tree(features=columns, filename="output/bin_tree", show_effect=True)