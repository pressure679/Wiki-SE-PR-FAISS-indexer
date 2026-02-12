import pandas as pd
import xgboost as xgb
wiki = "wikisimple"
"""
Root articles: Psychology, Religion, Computer science, Information technology, Data science, Day trading, Business, Economics
"""
def main():
    cols = [
        "query_id",
        "semantic",
        "pagerank",
        "is_lead",
        "length",
        "depth",
        "outlinks",
        "label",
    ]

    df = pd.read_csv("Wikipedia and StackExchange/" + wiki + "_xgb_train.csv", names=cols)
    df = df.dropna(subset=["label"])

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=3,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist"
    )

    model.fit(X, y)
    model.save_model("Wikipedia and StackExchange/" + wiki + "_xgb_ranker.json")

main()