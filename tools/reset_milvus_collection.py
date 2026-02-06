from __future__ import annotations

"""CLI utility to drop and recreate the Milvus collection."""

import argparse

from src.app.settings import settings


def main() -> None:
    """Reset the configured Milvus collection using app settings."""
    parser = argparse.ArgumentParser(description="Drop and recreate Milvus collection.")
    parser.add_argument(
        "--collection",
        default=settings.milvus_collection,
        help="Collection name to reset.",
    )
    args = parser.parse_args()

    try:
        from pymilvus import connections, utility
    except ImportError as exc:
        raise SystemExit("pymilvus is required to reset the collection") from exc

    connections.connect(alias="default", uri=settings.milvus_uri, token=settings.milvus_token)

    if utility.has_collection(args.collection):
        print(f"Dropping collection: {args.collection}")
        utility.drop_collection(args.collection)

    # Recreate by importing build path
    from src.app.dependencies import get_pipeline, reset_pipeline_cache

    reset_pipeline_cache()
    _ = get_pipeline()  # triggers collection creation
    print(f"Recreated collection: {args.collection}")


if __name__ == "__main__":
    main()
