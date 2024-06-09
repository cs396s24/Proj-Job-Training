import pandas as pd
import numpy as np
from CTL.causal_tree_learn import CausalTree
from IPython.display import Image

lalonde_psid = pd.read_csv("synthetic_observed.csv")
lalonde_psid["diff"] = lalonde_psid["re78"] - lalonde_psid["re75"]
lalonde_psid.drop(["nodegree"], axis=1, inplace=True)
treatment = lalonde_psid["treat"].values

y = lalonde_psid["diff"].values
x = lalonde_psid.drop(["treat", "re78", "diff"], axis=1).values
columns = lalonde_psid.drop(["treat", "re78"], axis=1).columns

def makeTree(x, y, treatment):
    ctl = CausalTree(magnitude=False)
    ctl.fit(x, y, treatment)
    ctl.prune()
    ctl.plot_tree(features=columns, filename="bin_tree_observed", show_effect=True)

makeTree(x, y, treatment)
img = Image(filename="bin_tree_exp.png", width=700)
display(img)

