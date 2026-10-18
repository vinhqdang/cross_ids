"""Cross-dataset flow-based intrusion detection: schema, datasets, RIFT and baselines."""

__version__ = "0.1.0"

try:  # pyarrow must load before torch on Windows, otherwise read_parquet segfaults
    import pyarrow.parquet  # noqa: F401
except ImportError:
    pass
