from __future__ import annotations

import abc
import enum
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel

if TYPE_CHECKING:
    from repipe.asset.asset import Asset


class StorageType(enum.StrEnum):
    CSV = "CSV"
    PARQUET = "PARQUET"
    ARROW = "ARROW"
    DUCKLAKE = "DUCKLAKE"


# TODO: some type of metadata to specify the hive partitioning of parquet

class BaseStorageConfig(BaseModel):
    """ Everything inheriting StorageConfig must have optionals as the other fields """
    pass


class ParquetStorageConfig(BaseStorageConfig):
    pass


class ArrowStorageConfig(BaseStorageConfig):
    pass


class CsvStorageConfig(BaseStorageConfig):
    pass


type StorageConfig = ParquetStorageConfig | ArrowStorageConfig | CsvStorageConfig


def make_default_storage_config(storage_type: StorageType) -> StorageConfig:
    match storage_type:  # TODO - unit test & enumerate over all
        case StorageType.PARQUET:
            return ParquetStorageConfig()
        case StorageType.ARROW:
            return ArrowStorageConfig()
        case StorageType.CSV:
            return CsvStorageConfig()
        case _:
            raise NotImplementedError()


class LoadStrategy[T, ConfigType](abc.ABC):

    @abstractmethod
    def load(self, config: ConfigType, asset: Asset) -> T:
        pass


class SaveStrategy[T, ConfigType](abc.ABC):

    @abstractmethod
    def save(self, config: ConfigType, asset: Asset, obj: T):
        pass


class StorageConcrete[LoadType, SaveType, ConfigType](
    LoadStrategy[LoadType, ConfigType], SaveStrategy[SaveType, ConfigType]
):

    @property
    @abstractmethod
    def config_type(self) -> type[ConfigType]:
        raise NotImplementedError()


# PARQUET
class ParquetPolars(StorageConcrete[pl.LazyFrame, pl.DataFrame | pl.LazyFrame, ParquetStorageConfig]):

    def save(self, config: ParquetStorageConfig, asset: Asset, obj: pl.DataFrame | pl.LazyFrame):
        path = self.make_path(config, asset)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, pl.DataFrame):
            obj.write_parquet(path)
        elif isinstance(obj, pl.LazyFrame):
            obj.sink_parquet(path)

    def load(self, config: ParquetStorageConfig, asset: Asset) -> pl.LazyFrame:
        path = self.make_path(config, asset)
        return pl.scan_parquet(path)

    @property
    def config_type(self) -> type[ParquetStorageConfig]:
        return ParquetStorageConfig

    def make_path(self, config: ParquetStorageConfig, asset: Asset) -> Path:
        return asset.base_storage_path_with_schema / f'{asset.name}.parquet'  # TODO is this the right extension?


# ARROW

class ArrowPolars(StorageConcrete[pl.LazyFrame, pl.DataFrame | pl.LazyFrame, ArrowStorageConfig]):

    def save(self, config: ArrowStorageConfig, asset: Asset, obj: pl.DataFrame | pl.LazyFrame):
        path = self.make_path(config, asset)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, pl.DataFrame):
            obj.write_ipc(path)
        elif isinstance(obj, pl.LazyFrame):
            obj.sink_ipc(path)

    def load(self, config: ArrowStorageConfig, asset: Asset) -> pl.LazyFrame:
        path = self.make_path(config, asset)
        return pl.scan_ipc(path)

    @property
    def config_type(self) -> type[ArrowStorageConfig]:
        return ArrowStorageConfig

    def make_path(self, config: ArrowStorageConfig, asset: Asset) -> Path:
        return asset.base_storage_path_with_schema / f'{asset.name}.arrow'  # TODO is this the right extension?


# CSV
class CsvPolars(StorageConcrete[pl.LazyFrame, pl.DataFrame | pl.LazyFrame, CsvStorageConfig]):

    def save(self, config: CsvStorageConfig, asset: Asset, obj: pl.DataFrame | pl.LazyFrame):
        path = self.make_path(config, asset)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, pl.DataFrame):
            obj.write_csv(path)
        elif isinstance(obj, pl.LazyFrame):
            obj.sink_csv(path)

    def load(self, config: CsvStorageConfig, asset: Asset) -> pl.LazyFrame:
        path = self.make_path(config, asset)
        return pl.scan_csv(path)

    @property
    def config_type(self) -> type[CsvStorageConfig]:
        return CsvStorageConfig

    def make_path(self, config: CsvStorageConfig, asset: Asset) -> Path:
        return asset.base_storage_path_with_schema / f'{asset.name}.csv'  # TODO is this the right extension?


# DUCKLAKE
# TODO

# Constants

STORAGE_TYPE_MAPPINGS = {
    StorageType.PARQUET: ParquetPolars,
    StorageType.ARROW: ArrowPolars,
    StorageType.CSV: CsvPolars,
}

CONFIG_TYPE_MAPPINGS = {
    ParquetStorageConfig: StorageType.PARQUET,
    ArrowStorageConfig: StorageType.ARROW,
    CsvStorageConfig: StorageType.CSV,
}


def get_storage_concrete(config: StorageConfig) -> StorageConcrete:
    """ Resolve the StorageConcrete implementation that handles this config. """
    storage_type = CONFIG_TYPE_MAPPINGS[type(config)]
    return STORAGE_TYPE_MAPPINGS[storage_type]()
