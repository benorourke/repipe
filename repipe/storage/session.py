from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from repipe.asset.partition import PartitionKey
from repipe.storage.storage import Mode, get_storage_concrete, staging_dir

if TYPE_CHECKING:
    from repipe.asset.asset import Asset


class WriteSession:
    """ Collects one run of an asset function's writes in a staging directory, so nothing is
    visible until `commit` - a run that fails part way leaves the previous output untouched. """

    def __init__(self, assets: Sequence[Asset], run_id: str, is_multi: bool):
        self.assets = {asset.name: asset for asset in assets}
        self.run_id = run_id
        self.is_multi = is_multi
        self._chunks: dict[str, int] = {}
        self._partitions: dict[str, set[PartitionKey]] = {}
        self._existing: dict[str, list[PartitionKey]] = {}

    @property
    def root(self) -> Path:
        return staging_dir(next(iter(self.assets.values())).base_storage_path, self.run_id)

    def stage(self, asset_name: str, data: Any, partition: Mapping[str, Any] | None = None) -> None:
        asset = self.assets[asset_name]
        storage = get_storage_concrete(asset.settings.storage)
        if asset_name not in self._chunks:
            self._check_layout(asset)
        self._chunks[asset_name] = self._chunks.get(asset_name, 0) + 1

        written = storage.stage(
            asset.settings.storage, asset, data, self.root / asset_name, self._chunks[asset_name], partition,
        )
        partitions = self._partitions.setdefault(asset_name, set())
        partitions |= written
        if written:
            self._check_partitions(asset, partitions)

    def commit(self) -> None:
        for asset in self.assets.values():
            if asset.name not in self._chunks and asset.settings.is_required and self.is_multi:
                raise ValueError(f"asset function didn't write its required output {asset.name!r}")

        for asset in self.assets.values():
            # An unwritten output of a multi-asset is left alone. A single asset that wrote
            # nothing still commits, so OVERWRITE leaves it empty, as it always has.
            if asset.name in self._chunks or not self.is_multi:
                self._commit(asset)
        self.abort()

    def abort(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _commit(self, asset: Asset) -> None:
        config = asset.settings.storage
        storage = get_storage_concrete(config)
        partitions = self._partitions.get(asset.name, set())
        settings = asset.settings.partition

        storage.promote(asset, self.root / asset.name, self.run_id)
        if config.mode == Mode.OVERWRITE:
            if settings is None or not settings.overwrite_by:
                storage.replace(asset, self.run_id)
            else:
                for unit in {key.prefix(len(settings.overwrite_by)) for key in partitions}:
                    storage.replace(asset, self.run_id, unit)
        elif config.mode == Mode.APPEND_THEN_MERGE:
            if settings is None:
                storage.consolidate(config, asset)
            for key in partitions:
                storage.consolidate(config, asset, key)

    def _check_layout(self, asset: Asset) -> None:
        settings = asset.settings.partition
        replaces_everything = asset.settings.storage.mode == Mode.OVERWRITE and (settings is None or not settings.overwrite_by)
        if not replaces_everything and not get_storage_concrete(asset.settings.storage).layout_matches(asset):
            raise ValueError(
                f"asset {asset.name!r} has existing data laid out for different partition settings - "
                f"run `repipe clear {asset.name}` first"
            )

    def _check_partitions(self, asset: Asset, partitions: set[PartitionKey]) -> None:
        settings = asset.settings.partition
        assert settings is not None
        if len(partitions) > settings.max_partitions_per_run:
            raise ValueError(
                f"asset {asset.name!r} wrote more than max_partitions_per_run={settings.max_partitions_per_run} "
                f"partitions - check it's partitioned by the right columns, or raise the limit"
            )

        # On a case-insensitive filesystem these would share a directory and lose rows.
        if asset.name not in self._existing:
            self._existing[asset.name] = get_storage_concrete(asset.settings.storage).existing_partitions(asset)
        seen: dict[str, PartitionKey] = {}
        for key in [*self._existing[asset.name], *partitions]:
            folded = str(key.relative_path()).casefold()
            if folded in seen and seen[folded] != key:
                raise ValueError(
                    f"asset {asset.name!r} has partitions {dict(seen[folded])!r} and {dict(key)!r}, "
                    f"which differ only by case"
                )
            seen[folded] = key
