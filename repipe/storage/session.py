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
    """ Commits each write into its asset's directory as soon as it's staged - a chunk is durable
    and visible the moment it's yielded, so a run that fails part way keeps everything it already
    wrote rather than losing it. """

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
        config = asset.settings.storage
        storage = get_storage_concrete(config)
        directory = self.root / asset_name
        if asset_name not in self._chunks:
            self._check_layout(asset)
            # Made up front, even if `data` is empty, so a missing directory means staging was deleted
            directory.mkdir(parents=True, exist_ok=True)
        self._check_staging_exists(asset_name)
        self._chunks[asset_name] = self._chunks.get(asset_name, 0) + 1

        written = storage.stage(config, asset, data, directory, self._chunks[asset_name], partition)
        partitions = self._partitions.setdefault(asset_name, set())
        partitions |= written
        if written:
            self._check_case_clashes(asset, partitions)

        storage.commit(config, asset, directory, self.run_id, written)

    def finish(self) -> None:
        """ Called once the asset function is done yielding. A required output of a multi-asset
        that never received a write fails the run - whatever other outputs it did write are
        already durable regardless. A single (non-multi) asset that wrote nothing still commits
        empty, so OVERWRITE leaves it empty, as it always has. """
        for asset in self.assets.values():
            if asset.name in self._chunks:
                continue
            if self.is_multi:
                if asset.settings.is_required:
                    raise ValueError(f"asset function didn't write its required output {asset.name!r}")
                continue
            config = asset.settings.storage
            get_storage_concrete(config).commit(config, asset, self.root / asset.name, self.run_id, set())
        self.abort()

    def abort(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _check_staging_exists(self, asset_name: str) -> None:
        # Otherwise what was staged is lost, and OVERWRITE would replace the asset with what's left
        if not (self.root / asset_name).is_dir():
            raise RuntimeError(f"staging for run {self.run_id} was deleted while it was running")

    def _check_layout(self, asset: Asset) -> None:
        settings = asset.settings.partition
        # A partitioned OVERWRITE only ever replaces the partitions it wrote to, so an
        # existing mismatched layout elsewhere in the asset would otherwise go unnoticed.
        replaces_everything = settings is None and asset.settings.storage.mode == Mode.OVERWRITE
        if not replaces_everything and not get_storage_concrete(asset.settings.storage).layout_matches(asset):
            raise ValueError(
                f"asset {asset.name!r} has existing data laid out for different partition settings - "
                f"run `repipe clear {asset.name}` first"
            )

    def _check_case_clashes(self, asset: Asset, partitions: set[PartitionKey]) -> None:
        """ On a case-insensitive filesystem these would share a directory and lose rows. """
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
