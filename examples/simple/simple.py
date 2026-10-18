import polars as pl
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris

from repipe import Repipe
from repipe.asset import AssetContext, MultiAssetOut, Output, PartitionSettings, asset, multi_asset


@asset(schemas=["source"])
def iris_raw(context: AssetContext) -> pl.DataFrame:
    """ Step 1: add the iris dataset. """
    iris = load_iris()  # pyright's bundled sklearn stub mis-resolves this to the return_X_y=True overload
    return pl.DataFrame(iris.data, schema=iris.feature_names).with_columns(  # pyright: ignore[reportAttributeAccessIssue]
        pl.Series("target", iris.target)  # pyright: ignore[reportAttributeAccessIssue]
    )


@asset(
    dependencies=["iris_raw"],
    schemas=["intermediate"],
    partition=PartitionSettings(columns={"species": pl.Int64}),
)
def iris_renamed(context: AssetContext, iris_raw: pl.DataFrame) -> list[Output]:
    """ Step 2: rename the columns to something friendlier than sklearn's defaults, writing one
    partition per species - a partitioned write always needs its key given explicitly. """
    renamed = iris_raw.rename({
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "species",
    })
    return [
        Output(group, partition={"species": species})
        for (species,), group in renamed.partition_by("species", as_dict=True).items()
    ]


@multi_asset(
    outputs={
        "iris_train": MultiAssetOut(schemas=["intermediate"]),
        "iris_test": MultiAssetOut(schemas=["intermediate"]),
    },
    dependencies=["iris_renamed"],
)
def iris_split(context: AssetContext, iris_renamed: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """ Step 3: hold back a fifth of the rows for testing. """
    shuffled = iris_renamed.sample(fraction=1.0, shuffle=True, seed=0)
    train_rows = len(shuffled) * 4 // 5
    return {"iris_train": shuffled.head(train_rows), "iris_test": shuffled.tail(-train_rows)}


@asset(
    dependencies=["iris_train"],
    schemas=["mart"],
)
def iris_model(context: AssetContext, iris_train: pl.DataFrame) -> LGBMClassifier:
    """ Step 4: train an LGBM model on the training rows. """
    model = LGBMClassifier(n_estimators=10, verbose=-1)
    model.fit(
        iris_train.select("sepal_length", "sepal_width", "petal_length", "petal_width"),
        iris_train["species"],
    )
    return model


app = Repipe(
    assets=[
        iris_raw,
        iris_renamed,
        iris_split,
        iris_model,
    ]
)
