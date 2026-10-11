# repipe

Framework for quantitative research &amp; data pipelining.

## Defining assets

Assets are plain Python functions decorated with `@asset`, returning a `polars` frame (or an ML model):

```python
import polars as pl
from repipe.asset.asset import asset, AssetContext


@asset
def raw_data(context: AssetContext) -> pl.LazyFrame:
    return pl.DataFrame({"a": [1, 2, 3]}).lazy()


@asset(dependencies=["raw_data"])
def transformed(context: AssetContext) -> pl.LazyFrame:
    ...
```

## Setting up a project

`repipe` isn't published, so add it as a local editable dependency to your own `uv` project:

```bash
uv init my_project
cd my_project
uv add --editable /path/to/repipe
```

Then point `repipe` at the file that defines your assets, in `pyproject.toml`:

```toml
[tool.repipe]
main = "my_pipeline.py"
```

`repipe run` walks up from your current directory to find the nearest `pyproject.toml`, so it works from any
subdirectory of your project, not just the root.

## Running

```bash
uv run repipe run                        # everything, in dependency order
uv run repipe run my_asset                # just my_asset
uv run repipe run +my_asset                # my_asset and everything upstream of it
uv run repipe run my_asset+                # my_asset and everything downstream of it
uv run repipe run '!my_asset'              # everything except my_asset (quote the `!`)
uv run repipe run '!my_asset+'              # everything except my_asset and its downstream (quote the `!`)
uv run repipe run label:nightly            # everything labeled "nightly"]
uv run repipe test my_test                 # this test must not apply to several assets
uv run repipe test my_asset::my_test       # a test, qualified by the asset it's attached to (use when a test applies to several assets)
```

## Examples

[`examples/simple`](examples/simple)
