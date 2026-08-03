#!/usr/bin/env python3
"""Export the MITOMAP Genome Loci web table as a clean, versioned TSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_SOURCE_URL = (
    "https://www.mitomap.org/foswiki/bin/view/MITOMAP/GenomeLoci"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "mitomap"
)
GENOME_LENGTH = 16_569
EXPORT_COLUMNS = [
    "map_locus",
    "start",
    "end",
    "shorthand",
    "description",
    "reference_count",
    "reference_ids",
    "reference_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download MITOMAP Genome Loci and export the JavaScript-backed "
            "table to a versioned TSV plus a JSON provenance record."
        )
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help=f"MITOMAP page URL (default: {DEFAULT_SOURCE_URL})",
    )
    parser.add_argument(
        "--input-html",
        type=Path,
        help=(
            "Parse an existing HTML file instead of downloading the page. "
            "Useful for offline reproduction and parser tests."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--expected-revision",
        help="Fail unless the page revision matches this value, e.g. r889.",
    )
    parser.add_argument(
        "--expected-tsv-sha256",
        help=(
            "Fail unless the normalized TSV has this SHA256 checksum. "
            "This is stable even when non-table HTML changes between requests."
        ),
    )
    parser.add_argument(
        "--expected-record-count",
        type=int,
        help="Fail unless the exported table has exactly this many loci.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Download timeout in seconds (default: 60).",
    )
    return parser.parse_args()


def obtain_html(args: argparse.Namespace) -> bytes:
    if args.input_html is not None:
        source_path = args.input_html.resolve()
        return source_path.read_bytes()

    request = Request(
        args.source_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; mtdna_constraint MITOMAP exporter)"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urlopen(request, timeout=args.timeout) as response:
        return response.read()


def extract_page_metadata(page_text: str) -> dict[str, str]:
    revision_matches = re.findall(
        r"Topic revision:\s*r?(\d+)\s*-\s*([^,<]+)",
        page_text,
    )
    if not revision_matches:
        raise ValueError("Could not identify the MITOMAP topic revision.")
    revision_number, revision_date = revision_matches[-1]

    last_edited_match = re.search(r"Last Edited:\s*([^<\r\n]+)", page_text)
    if not last_edited_match:
        raise ValueError("Could not identify the MITOMAP Last Edited date.")

    return {
        "source_revision": f"r{revision_number}",
        "source_revision_date": revision_date.strip(),
        "source_last_edited": last_edited_match.group(1).strip(),
    }


def extract_table_payload(page_text: str) -> list[list[object]]:
    data_match = re.search(
        r'"data"\s*:\s*(\[.*\])\s*,\s*"columns"\s*:',
        page_text,
        flags=re.DOTALL,
    )
    if not data_match:
        raise ValueError("Could not locate the MITOMAP DataTables data array.")

    payload = json.loads(data_match.group(1))
    if not isinstance(payload, list) or not payload:
        raise ValueError("MITOMAP returned an empty or malformed locus table.")
    if any(not isinstance(row, list) or len(row) != 6 for row in payload):
        raise ValueError("Expected six fields in every MITOMAP locus row.")
    return payload


def parse_reference(
    reference_html: object,
    source_url: str,
) -> tuple[int, str, str]:
    reference_text = html.unescape(str(reference_html)).strip()
    link_match = re.search(r'href="([^"]+)"', reference_text)
    count_match = re.search(r">(\d+)</a>", reference_text)

    if link_match:
        reference_url = urljoin(source_url, link_match.group(1))
        query = parse_qs(urlparse(reference_url).query)
        reference_ids = query.get("refs", [""])[0]
        if not count_match:
            raise ValueError(
                f"Reference link lacks a numeric count: {reference_text}"
            )
        reference_count = int(count_match.group(1))
    else:
        reference_url = ""
        reference_ids = ""
        try:
            reference_count = int(reference_text)
        except ValueError as error:
            raise ValueError(
                f"Unrecognized MITOMAP reference field: {reference_text}"
            ) from error

    if reference_ids:
        parsed_count = len([value for value in reference_ids.split(",") if value])
        if parsed_count != reference_count:
            raise ValueError(
                "MITOMAP reference count disagrees with the linked reference IDs."
            )

    return reference_count, reference_ids, reference_url


def normalize_rows(
    table_payload: list[list[object]],
    source_url: str,
) -> list[dict[str, object]]:
    records = []
    for row in table_payload:
        map_locus, start, end, shorthand, description, reference_html = row
        start = int(start)
        end = int(end)
        if not 1 <= start <= GENOME_LENGTH or not 1 <= end <= GENOME_LENGTH:
            raise ValueError(
                f"Coordinates outside rCRS for {map_locus}: {start}-{end}"
            )
        reference_count, reference_ids, reference_url = parse_reference(
            reference_html,
            source_url,
        )
        records.append({
            "map_locus": str(map_locus).strip(),
            "start": start,
            "end": end,
            "shorthand": str(shorthand).strip(),
            "description": str(description).strip(),
            "reference_count": reference_count,
            "reference_ids": reference_ids,
            "reference_url": reference_url,
        })

    locus_names = [record["map_locus"] for record in records]
    if len(set(locus_names)) != len(locus_names):
        raise ValueError("MITOMAP export contains duplicate locus names.")
    required_text_fields = ["map_locus", "shorthand", "description"]
    if any(
        not str(record[field]).strip()
        for record in records
        for field in required_text_fields
    ):
        raise ValueError("MITOMAP export contains empty required fields.")

    control_region = [
        record for record in records if record["map_locus"] == "MT-CR"
    ]
    if (
        len(control_region) != 1
        or control_region[0]["start"] != 16_024
        or control_region[0]["end"] != 576
    ):
        raise ValueError("Expected circular MT-CR coordinates 16024-576.")
    return records


def write_tsv(
    records: list[dict[str, object]],
    output_path: Path,
    expected_sha256: str | None = None,
) -> str:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=EXPORT_COLUMNS,
            dialect="excel-tab",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    tsv_sha256 = sha256_bytes(temporary_path.read_bytes())
    if expected_sha256 and tsv_sha256 != expected_sha256.lower():
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"Expected TSV SHA256 {expected_sha256.lower()}, "
            f"found {tsv_sha256}."
        )
    temporary_path.replace(output_path)
    return tsv_sha256


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    page_bytes = obtain_html(args)
    page_text = page_bytes.decode("utf-8")
    page_metadata = extract_page_metadata(page_text)

    if (
        args.expected_revision
        and page_metadata["source_revision"] != args.expected_revision
    ):
        raise ValueError(
            f"Expected {args.expected_revision}, found "
            f"{page_metadata['source_revision']}."
        )

    records = normalize_rows(
        extract_table_payload(page_text),
        args.source_url,
    )
    if (
        args.expected_record_count is not None
        and len(records) != args.expected_record_count
    ):
        raise ValueError(
            f"Expected {args.expected_record_count} loci, found {len(records)}."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    revision = page_metadata["source_revision"]
    output_tsv = args.output_dir / f"mitomap_genome_loci_192_{revision}.tsv"
    output_metadata = (
        args.output_dir / f"mitomap_genome_loci_192_{revision}.metadata.json"
    )
    tsv_sha256 = write_tsv(
        records,
        output_tsv,
        expected_sha256=args.expected_tsv_sha256,
    )

    metadata = {
        "exporter": "scripts/export_mitomap_functional_loci_192.py",
        "exporter_version": 1,
        "parser_schema_version": 1,
        "source_url": args.source_url,
        **page_metadata,
        "genome_reference": "rCRS / NC_012920.1",
        "genome_length": GENOME_LENGTH,
        "coordinate_convention": (
            "1-based inclusive; circular intervals may have start > end"
        ),
        "record_count": len(records),
        "tsv_file": output_tsv.name,
        "tsv_sha256": tsv_sha256,
    }
    temporary_metadata = output_metadata.with_suffix(
        output_metadata.suffix + ".tmp"
    )
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(output_metadata)

    print(f"Exported {len(records)} MITOMAP loci ({revision}).")
    print(f"TSV: {output_tsv}")
    print(f"Metadata: {output_metadata}")


if __name__ == "__main__":
    main()
