import importlib.util
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

from frozendict import frozendict
from pydantic import BaseModel

from repipe.asset.asset import Asset, AssetFunction, RegisteredAssetFunction
from repipe.storage import get_storage_concrete


class Repipe:
    assets: Mapping[str, Asset]

    def __init__(
            self,
            assets: list[Callable],
            base_storage_path: Path | None = None,
    ):

        if base_storage_path is None:
            base_storage_path = Path.cwd() / "data"

        Repipe._validate_assets_are_decorated(assets)
        self.assets = frozendict(Repipe.construct_assets_from_funcs(assets, base_storage_path))

    @staticmethod
    def construct_assets_from_funcs(
            asset_functions: list[Callable],
            base_storage_path: Path
    ) -> dict[str, Asset]:
        assets: dict[str, Asset] = {}
        for asset_func in asset_functions:
            registered: RegisteredAssetFunction = getattr(asset_func, "__asset__")
            function = AssetFunction(
                name=registered.name,
                func=registered.func,
                dependencies=registered.dependencies,
                outputs=list(registered.outputs),
                is_multi=registered.is_multi,
                python_func=registered.func.__name__,
                python_module=asset_func.__module__,
                python_file=sys.modules[asset_func.__module__].__file__ or "",
            )
            for asset_name, settings in registered.outputs.items():
                if asset_name in assets:
                    raise ValueError(f"two assets are named {asset_name!r} - asset names must be unique")
                assets[asset_name] = Asset(
                    name=asset_name,
                    base_storage_path=base_storage_path,
                    settings=settings,
                    function=function,
                )
        return assets

    @staticmethod
    def _validate_assets_are_decorated(assets: list[Callable]) -> None:
        undecorated = [a for a in assets if not hasattr(a, '__asset__')]
        if undecorated:
            names = [getattr(a, '__name__', repr(a)) for a in undecorated]
            raise ValueError(f"Not decorated with @asset: {names}")


class RepipeConfig(BaseModel):
    """ [tool.repipe] table in pyproject.toml """

    main: str
    """ Path (relative to the project root) to the python file defining all assets. """


def find_project_root(start: Path | None = None) -> Path:
    """ Walks upward from start, defaults to CWD, until a pyproject.toml is found. """
    current = (start or Path.cwd()).resolve()

    for directory in (current, *current.parents):
        if (directory / "pyproject.toml").is_file():
            return directory

    raise FileNotFoundError(f"No pyproject.toml found in {current} or any parent directory")


def load_project_config(project_root: Path | None = None) -> tuple[Path, RepipeConfig]:
    """ Locates pyproject.toml and parses its [tool.repipe] table. """
    root = project_root or find_project_root()
    pyproject_path = root / "pyproject.toml"

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    tool_config = data.get("tool", {}).get("repipe")
    if tool_config is None:
        raise ValueError(
            f"{pyproject_path} has no [tool.repipe] table. Add e.g.:\n\n"
            f'[tool.repipe]\nmain = "examples/simple/simple.py"\n'
        )

    return root, RepipeConfig(**tool_config)


def load_main_module(project_root: Path, config: RepipeConfig) -> ModuleType:
    """ Imports the project's main file by path """
    main_path = (project_root / config.main).resolve()

    if not main_path.is_file():
        raise FileNotFoundError(f"repipe main file not found: {main_path}")

    module_name = f"repipe._main.{main_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load a module spec for {main_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def load_app(project_root: Path | None = None) -> tuple[Path, ModuleType, Repipe]:
    """ Locates the project, imports its main file, and returns its single Repipe object. """
    root, config = load_project_config(project_root)
    module = load_main_module(root, config)

    apps = [v for v in vars(module).values() if isinstance(v, Repipe)]
    if not apps:
        raise ValueError(
            f"{module.__file__} does not define a Repipe object. "
            f"Add e.g. `app = Repipe(assets=[my_asset, ...])`."
        )
    if len(apps) > 1:
        raise ValueError(f"{module.__file__} defines multiple Repipe objects - expected exactly one.")

    return root, module, apps[0]


def load(*names: str) -> Any:
    """ Loads one or more assets by name. """
    if not names:
        raise ValueError("load() needs at least one asset name")

    _, _, app = load_app()
    values = tuple(_load_one(app, name) for name in names)
    return values[0] if len(values) == 1 else values


def _load_one(app: Repipe, name: str) -> Any:
    asset = app.assets.get(name)
    if asset is None:
        raise KeyError(f"no asset named {name!r} - known assets: {sorted(app.assets)}")
    return get_storage_concrete(asset.settings.storage).load(asset.settings.storage, asset)
