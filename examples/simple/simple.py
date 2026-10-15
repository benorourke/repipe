import polars as pl
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris

from repipe import Repipe
from repipe.asset.asset import asset, AssetContext
from repipe.storage import LGBMStorageConfig


@asset(schemas=["source"])
def iris_raw(context: AssetContext) -> pl.DataFrame:
    """ Step 1: add the iris dataset. """
    iris = load_iris()  # pyright's bundled sklearn stub mis-resolves this to the return_X_y=True overload
    return pl.DataFrame(iris.data, schema=iris.feature_names).with_columns(  # pyright: ignore[reportAttributeAccessIssue]
        pl.Series("target", iris.target)  # pyright: ignore[reportAttributeAccessIssue]
    )


@asset(dependencies=["iris_raw"], schemas=["intermediate"])
def iris_renamed(context: AssetContext, iris_raw: pl.DataFrame) -> pl.DataFrame:
    """ Step 2: rename the columns to something friendlier than sklearn's defaults. """
    return iris_raw.rename({
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "species",
    })


@asset(
    dependencies=["iris_renamed"],
    schemas=["mart"],
    storage=LGBMStorageConfig(),
)
def iris_model(context: AssetContext, iris_renamed: pl.DataFrame) -> LGBMClassifier:
    """ Step 3: train an LGBM model on the renamed iris data. """
    model = LGBMClassifier(n_estimators=10, verbose=-1)
    model.fit(
        iris_renamed.select("sepal_length", "sepal_width", "petal_length", "petal_width"),
        iris_renamed["species"],
    )
    return model


app = Repipe(
    assets=[
        iris_raw,
        iris_renamed,
        iris_model,
    ]
)
