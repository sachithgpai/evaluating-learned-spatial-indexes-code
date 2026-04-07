import argparse

import numpy as np
import osmium
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_BATCH_SIZE = 500_000
DEFAULT_KEEP_EVERY = 10


class NodeParquetWriter(osmium.SimpleHandler):
    def __init__(self, out_path: str, batch_size: int, keep_every: int):
        super().__init__()
        self.out_path = out_path
        self.batch_size = batch_size
        self.keep_every = keep_every

        self.schema = pa.schema(
            [
                ("lat", pa.float32()),
                ("lon", pa.float32()),
            ]
        )
        self.writer = pq.ParquetWriter(
            out_path,
            self.schema,
            compression="zstd",
            use_dictionary=False,
        )

        self.lat = np.empty(self.batch_size, dtype=np.float32)
        self.lon = np.empty(self.batch_size, dtype=np.float32)
        self.buffer_index = 0
        self.node_counter = 0
        self.total_written = 0

    def node(self, node) -> None:
        if not node.location.valid():
            return

        self.node_counter += 1
        if self.node_counter % self.keep_every != 0:
            return

        self.lat[self.buffer_index] = node.location.lat
        self.lon[self.buffer_index] = node.location.lon
        self.buffer_index += 1

        if self.buffer_index >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self.buffer_index == 0:
            return

        table = pa.Table.from_arrays(
            [
                pa.array(self.lat[: self.buffer_index], type=pa.float32()),
                pa.array(self.lon[: self.buffer_index], type=pa.float32()),
            ],
            schema=self.schema,
        )
        self.writer.write_table(table)
        self.total_written += self.buffer_index
        print(f"Wrote {self.total_written} datapoints so far...", flush=True)
        self.buffer_index = 0

    def close(self) -> None:
        self.flush()
        self.writer.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an OSM PBF file into a parquet with lat/lon columns."
    )
    parser.add_argument("input_pbf", help="Path to the source .osm.pbf file.")
    parser.add_argument(
        "output_parquet",
        nargs="?",
        default="planet_latlon.parquet",
        help="Path to the output parquet file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="How many sampled nodes to buffer before writing a parquet batch.",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        default=DEFAULT_KEEP_EVERY,
        help="Keep every Nth valid node from the OSM stream.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.keep_every <= 0:
        raise ValueError("--keep-every must be positive.")

    handler = NodeParquetWriter(
        out_path=args.output_parquet,
        batch_size=args.batch_size,
        keep_every=args.keep_every,
    )
    handler.apply_file(args.input_pbf, locations=False)
    handler.close()
    print(f"There are {handler.total_written} datapoints.", flush=True)


if __name__ == "__main__":
    main()
