from utils.utilities import load_features_from_file
from functools import reduce

HEADER = """
\\begin{table}[h]
	\\centering 
 	\\caption{Selected feature sets of the \\ac{fds} models.}
	\\label{table:features}
    \\begin{tabular}{|p{5em} p{32em}|}
    \\hline
    \\textbf{Model} & \\textbf{Features} \\\\
    \\hline \\hline
"""

FOOTER = """
\\end{tabular}
\\end{table}
"""


def make_feature_row(model_name, features):
    features = [f.replace("_", "\\_") for f in features]
    features = ", ".join(features)
    features = "\\scriptsize{\\texttt{" + features + "}}"
    row = "\n%s & %s \\\\" % (model_name, features)
    row += "\n \\hline \\hline"
    return row


if __name__ == "__main__":
    table = HEADER
    models = [
        "logistic_regression",
        "neural_network",
        "random_forest",
        "support_vector_machine",
        "xgboost",
    ]
    model_names = {
        "random_forest": "\\ac{rf}",
        "logistic_regression": "\\ac{lr}",
        "neural_network": "\\ac{ann}",
        "support_vector_machine": "\\ac{svm}",
        "xgboost": "\\ac{xgb}",
    }

    feature_sets = []
    for m in models:
        feature_file = "config/models/features/%s.txt" % m
        feature_sets.append(set(load_features_from_file(feature_file)))

    common = list(reduce(lambda x, y: x.intersection(y), feature_sets))
    table += make_feature_row("SHARED", common)
    for i, m in enumerate(models):
        features = feature_sets[i].difference(set(common))
        model_name = model_names[m]
        table += make_feature_row(model_name, features)

    table += FOOTER
    with open("utils/features.tex", "w") as f:
        f.write(table)
