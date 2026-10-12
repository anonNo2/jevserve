"""Acquire the discovered baseline using the same safe preparer as future serving."""

import argparse
import json
from pathlib import Path

from jevserve.artifacts.manifest import ArtifactFile, ModelManifest
from jevserve.artifacts.prepare import prepare_model


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "models/locks/open-jev-2b.json").read_text())
    for name, source in [("checkpoint", lock), ("base", lock["base"])]:
        manifest = ModelManifest(
            model_id=source["model_id"],
            revision=source["revision"],
            files=tuple(
                ArtifactFile(
                    **item,
                    source_repo=source["model_id"],
                    source_revision=source["revision"],
                    source_path=item["path"],
                )
                for item in source["files"]
            ),
        )
        prepared = prepare_model(
            manifest, args.cache, offline=args.offline, token_file=args.token_file
        )
        print(f"{name}: {prepared.path}")


if __name__ == "__main__":
    main()
