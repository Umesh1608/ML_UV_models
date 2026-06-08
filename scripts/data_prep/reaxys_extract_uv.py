#!/usr/bin/env python3
"""
Reaxys API extraction of UV-Vis spectral data for molecules from Mamede et al. (2021).

Mamede et al. classified ~75K organic molecules from Reaxys as POS/NEG for
photoreactive potential (UV absorption in 290-700 nm with MEC > 1000). Their
supplementary data provides SMILES + Reaxys Registry Numbers (XRN) + class labels,
but NOT raw λ_max values.

This script extracts actual UV-Vis spectral data (λ_max, MEC, solvent) from Reaxys
for those molecules, enabling regression-based comparison.

Usage:
    python3 reaxys_extract_uv.py --test                       # debug run (5 XRNs)
    python3 reaxys_extract_uv.py                               # full extraction
    python3 reaxys_extract_uv.py --apikey YOUR_KEY             # pass API key via CLI
    python3 reaxys_extract_uv.py --input data/mamede_parsed.csv  # custom input
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REAXYS_API_URL = "https://www.reaxys.com/reaxys/api"
CALLER_ID = "uv_extract"
BATCH_SIZE = 50  # XRNs per search request (conservative to stay under timeout)
RETRIEVE_PAGE = 100  # results per retrieve call
SLEEP_BETWEEN_BATCHES = 1.5  # seconds
SESSION_TIMEOUT_ANON = 1700  # ~28 min (anonymous sessions expire at 30 min)
SESSION_TIMEOUT_AUTH = 21000  # ~5.8 hrs (authenticated sessions expire at 6 hrs)
DEBUG_XRNS = 5  # number of XRNs for --test mode

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PARSED_FILE = DATA_DIR / "mamede_parsed.csv"
RAW_UV_FILE = RAW_DIR / "reaxys_uv_raw.csv"
PROGRESS_FILE = RAW_DIR / "reaxys_progress.json"
DEBUG_XML_FILE = RAW_DIR / "debug_reaxys_response.xml"
SUPP_FILE = RAW_DIR / "mamede_supplementary.txt"

# Reaxys field identifiers — updated after --test debug inspection
# These are the standard Reaxys API field names; adjust if debug XML shows different tags
FIELDS_DEBUG = []  # empty = retrieve all fields (for discovery)
FIELDS_UV = [
    "IDE.XRN",
    "IDE.SMILES",
    "DAT.SPUV",       # UV spectral data group
    "DAT.SPUV.WAV",   # absorption wavelength (nm)
    "DAT.SPUV.EXT",   # molar extinction coefficient
    "DAT.SPUV.SOL",   # solvent
]


# ---------------------------------------------------------------------------
# Mamede supplementary file parser
# ---------------------------------------------------------------------------
def parse_mamede_supplementary(input_path: Path) -> list[dict]:
    """Parse the Mamede et al. supplementary file and extract XRN + SMILES.

    File format (tab-separated, 3 columns):
        REAXYS_Registry Number | SMILES | Substance Identification: Links to Reaxys

    Note: Class labels (POS/NEG) are NOT in this file — they were derived from
    Reaxys UV data. We reconstruct them in postprocess_reaxys.py after extraction.
    """
    records = []
    with open(input_path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        print(f"  Header: {header}")

        # Map column names
        col_map = {}
        for i, h in enumerate(header):
            h_lower = h.strip().lower()
            if "registry" in h_lower or "xrn" in h_lower:
                col_map["xrn"] = i
            elif "smiles" in h_lower:
                col_map["smiles"] = i

        if "xrn" not in col_map or "smiles" not in col_map:
            # Fallback: col 0 = XRN, col 1 = SMILES (matches observed format)
            print("  Using positional column mapping (col0=XRN, col1=SMILES)")
            col_map = {"xrn": 0, "smiles": 1}

        for row in reader:
            if not row or len(row) <= max(col_map.values()):
                continue
            xrn_raw = row[col_map["xrn"]].strip()
            xrn_clean = re.sub(r"[^\d]", "", xrn_raw)
            if not xrn_clean:
                continue
            records.append({
                "xrn": xrn_clean,
                "smiles": row[col_map["smiles"]].strip(),
            })

    return records


def save_parsed(records: list[dict], output_path: Path):
    """Save parsed Mamede data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "xrn"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved {len(records)} records to {output_path}")


def load_parsed(path: Path) -> list[dict]:
    """Load previously parsed CSV."""
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Reaxys API session management
# ---------------------------------------------------------------------------
class ReaxysSession:
    """Manages a Reaxys API session with auto-reconnect.

    Follows the official Reaxys API User Guide:
    - caller attribute = API key (application caller name)
    - X-els-apikey header sent with every request
    - Cookies (JSESSIONID, stationid, AWSELB) stored and returned automatically
    - Anonymous connect with empty username/password (IP-based auth)
    """

    def __init__(self, api_key: str | None = None, username: str = "", password: str = ""):
        self.api_key = api_key or ""
        self.username = username
        self.password = password
        self.session_id: str | None = None
        self.last_request_time: float = 0
        # requests.Session handles cookie persistence automatically
        self.http = requests.Session()
        self.http.headers.update({
            "Content-Type": "text/xml; charset=UTF-8",
            "Accept": "*/*",
        })
        if self.api_key:
            self.http.headers["X-els-apikey"] = self.api_key

    def _build_xml(self, inner: str, include_session: bool = True) -> str:
        """Build a Reaxys API XML request envelope."""
        parts = [f'caller="{self.api_key}"']
        if include_session and self.session_id:
            parts.append(f'sessionid="{self.session_id}"')
        attrs = " ".join(parts)
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f"<xf>\n"
            f"  <request {attrs}>\n"
            f"    {inner}\n"
            f"  </request>\n"
            f"</xf>"
        )

    def _post(self, xml_payload: str, timeout: int = 240) -> ET.Element:
        """Send POST request and parse XML response."""
        self.last_request_time = time.time()
        resp = self.http.post(REAXYS_API_URL, data=xml_payload, timeout=timeout)
        resp.raise_for_status()
        return ET.fromstring(resp.text)

    def _check_error(self, root: ET.Element) -> str | None:
        """Check for API error in response. Returns error message or None."""
        status = root.find(".//status")
        if status is not None and status.text and status.text.strip() == "ERROR":
            errnum = root.find(".//errnum")
            message = root.find(".//message")
            longtext = root.find(".//longtext")
            code = errnum.text.strip() if errnum is not None and errnum.text else "?"
            msg = message.text.strip() if message is not None and message.text else ""
            detail = longtext.text.strip() if longtext is not None and longtext.text else ""
            return f"Reaxys error {code}: {msg}" + (f" ({detail})" if detail else "")
        return None

    def connect(self):
        """Open a Reaxys API session."""
        inner = (
            f'<statement command="connect" '
            f'username="{self.username}" password="{self.password}"/>'
        )
        xml = self._build_xml(inner, include_session=False)

        try:
            root = self._post(xml)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot reach Reaxys API at {REAXYS_API_URL}.\n"
                f"  If you're off-campus, connect to your university VPN first.\n"
                f"  Error: {e}"
            ) from e

        err = self._check_error(root)
        if err:
            raise ConnectionError(
                f"Failed to connect: {err}\n\n"
                f"  Troubleshooting (try in order):\n"
                f"  A) Connect to university VPN and retry (IP-based auth)\n"
                f"  B) Register at https://www.reaxys.com (on VPN/campus),\n"
                f"     then use --username YOUR_EMAIL --password YOUR_PW\n"
                f"  C) Contact your library for the institutional API caller name"
            )

        sid = root.find(".//sessionid")
        if sid is None or not sid.text:
            raise ConnectionError(
                f"No sessionid in response. Full response:\n"
                f"{ET.tostring(root, encoding='unicode')}"
            )
        self.session_id = sid.text.strip()
        auth_type = "authenticated" if self.username else "anonymous/IP-based"
        timeout_min = self._session_timeout // 60
        print(f"  Connected to Reaxys ({auth_type}, "
              f"timeout ~{timeout_min} min, "
              f"session: {self.session_id[:16]}...)")

    def disconnect(self):
        """Close the Reaxys API session."""
        if not self.session_id:
            return
        # Per API docs: sessionid goes on <statement> for disconnect
        inner = f'<statement command="disconnect" sessionid="{self.session_id}"/>'
        xml = self._build_xml(inner, include_session=False)
        try:
            self._post(xml, timeout=30)
            print("  Disconnected from Reaxys.")
        except Exception as e:
            print(f"  Warning: disconnect failed: {e}")
        self.session_id = None

    @property
    def _session_timeout(self) -> int:
        """Return appropriate timeout based on whether credentials were provided."""
        return SESSION_TIMEOUT_AUTH if self.username else SESSION_TIMEOUT_ANON

    def ensure_session(self):
        """Reconnect if session may have timed out."""
        if not self.session_id:
            self.connect()
            return
        elapsed = time.time() - self.last_request_time
        if elapsed > self._session_timeout:
            print("  Session may have timed out, reconnecting...")
            self.connect()

    def search(self, xrn_list: list[str]) -> tuple[str, int]:
        """Search Reaxys by XRN list. Returns (resultname, result_count).

        Uses the official API format with <select_list>, <from_clause>,
        <where_clause>, and <options> nodes.
        """
        self.ensure_session()
        conditions = " OR ".join(f"IDE.XRN = {xrn}" for xrn in xrn_list)
        inner = (
            '<statement command="select"/>\n'
            "    <select_list><select_item/></select_list>\n"
            '    <from_clause dbname="RX" context="S">\n'
            f"      <where_clause>{conditions}</where_clause>\n"
            "    </from_clause>\n"
            "    <options>WORKER,NO_CORESULT</options>"
        )
        xml = self._build_xml(inner)
        root = self._post(xml)

        err = self._check_error(root)
        if err:
            if "1004" in err:
                print(f"  Session expired, reconnecting: {err}")
                self.connect()
                xml = self._build_xml(inner)
                root = self._post(xml)
                err = self._check_error(root)
                if err:
                    raise RuntimeError(err)
            else:
                raise RuntimeError(err)

        rname = root.find(".//resultname")
        rsize = root.find(".//resultsize")
        if rname is None:
            raise RuntimeError(
                f"No resultname in search response:\n"
                f"{ET.tostring(root, encoding='unicode')}"
            )
        result_name = rname.text.strip()
        result_size = int(rsize.text.strip()) if rsize is not None else 0
        return result_name, result_size

    def retrieve(
        self,
        result_name: str,
        first: int,
        last: int,
        fields: list[str] | None = None,
    ) -> ET.Element:
        """Retrieve results from a previous search.

        Uses the official API format with <select_list> and <from_clause>.
        """
        self.ensure_session()
        select_items = "<select_list><select_item/></select_list>"
        if fields:
            select_items = "<select_list>" + "".join(
                f"<select_item>{f}</select_item>" for f in fields
            ) + "</select_list>"
        inner = (
            '<statement command="select"/>\n'
            f"    {select_items}\n"
            f'    <from_clause resultname="{result_name}" '
            f'first_item="{first}" last_item="{last}"/>'
        )
        xml = self._build_xml(inner)
        root = self._post(xml)

        err = self._check_error(root)
        if err:
            if "1004" in err:
                print(f"  Session error on retrieve, reconnecting: {err}")
                self.connect()
                raise SessionExpiredError(err)
            raise RuntimeError(err)
        return root

    def retrieve_all_raw_xml(
        self,
        result_name: str,
        result_size: int,
        fields: list[str] | None = None,
    ) -> str:
        """Retrieve all results and return raw XML string (for debug)."""
        self.ensure_session()
        last = min(result_size, RETRIEVE_PAGE)
        select_items = "<select_list><select_item/></select_list>"
        if fields:
            select_items = "<select_list>" + "".join(
                f"<select_item>{f}</select_item>" for f in fields
            ) + "</select_list>"
        inner = (
            '<statement command="select"/>\n'
            f"    {select_items}\n"
            f'    <from_clause resultname="{result_name}" '
            f'first_item="1" last_item="{last}"/>'
        )
        xml = self._build_xml(inner)
        resp = self.http.post(REAXYS_API_URL, data=xml, timeout=240)
        resp.raise_for_status()
        return resp.text


class SessionExpiredError(Exception):
    pass


# ---------------------------------------------------------------------------
# UV data extraction from XML response
# ---------------------------------------------------------------------------
def extract_uv_records(root: ET.Element) -> list[dict]:
    """Extract UV-Vis records from a Reaxys retrieve response.

    This parser handles the typical Reaxys XML structure. The exact tag names
    may vary — run --test first to confirm.

    Returns a list of dicts with keys: xrn, smiles, wavelength_nm, mec, solvent
    """
    records = []

    # Reaxys substances are typically under <substance> or <hit> elements
    for substance in _iter_substances(root):
        xrn = _get_text(substance, ".//IDE.XRN") or _get_text(substance, ".//XRN") or ""
        smiles = (
            _get_text(substance, ".//IDE.SMILES")
            or _get_text(substance, ".//YY.STR")
            or _get_text(substance, ".//SMILES")
            or ""
        )

        # Find UV spectral data groups
        for spuv in _iter_uv_groups(substance):
            wav_text = (
                _get_text(spuv, ".//DAT.SPUV.WAV")
                or _get_text(spuv, ".//WAV")
                or _get_text(spuv, ".//WAVELENGTH")
                or _get_text(spuv, ".//absorption_maximum")
                or ""
            )
            mec_text = (
                _get_text(spuv, ".//DAT.SPUV.EXT")
                or _get_text(spuv, ".//EXT")
                or _get_text(spuv, ".//EXTINCTION")
                or _get_text(spuv, ".//molar_extinction")
                or ""
            )
            sol_text = (
                _get_text(spuv, ".//DAT.SPUV.SOL")
                or _get_text(spuv, ".//SOL")
                or _get_text(spuv, ".//SOLVENT")
                or _get_text(spuv, ".//solvent")
                or ""
            )

            # Parse wavelength — may be a range like "350-360" or a single value
            wavelengths = _parse_wavelength(wav_text)
            mec = _parse_float(mec_text)

            for wl in wavelengths:
                records.append({
                    "xrn": xrn.strip(),
                    "smiles": smiles.strip(),
                    "wavelength_nm": wl,
                    "mec": mec,
                    "solvent": sol_text.strip(),
                })

        # If no UV-specific group found, try generic spectral data
        if not list(_iter_uv_groups(substance)):
            # Check for any wavelength-like fields at substance level
            for tag_path in [".//absorption_maximum", ".//WAV", ".//WAVELENGTH",
                             ".//DAT.SPUV.WAV", ".//lambda_max"]:
                el = substance.find(tag_path)
                if el is not None and el.text:
                    for wl in _parse_wavelength(el.text):
                        records.append({
                            "xrn": xrn.strip(),
                            "smiles": smiles.strip(),
                            "wavelength_nm": wl,
                            "mec": None,
                            "solvent": "",
                        })

    return records


def _iter_substances(root: ET.Element):
    """Iterate over substance/hit elements in various Reaxys XML formats."""
    # Try common container tag names
    for tag in ["substance", "hit", "record", "item", "SUBSTANCE", "HIT"]:
        found = root.findall(f".//{tag}")
        if found:
            yield from found
            return
    # Fallback: yield root itself if it has XRN-like children
    if root.find(".//IDE.XRN") is not None or root.find(".//XRN") is not None:
        yield root


def _iter_uv_groups(substance: ET.Element):
    """Iterate over UV spectral data groups within a substance."""
    for tag in ["DAT.SPUV", "SPUV", "UV_VIS", "uv_vis", "spectral_data"]:
        found = substance.findall(f".//{tag}")
        if found:
            yield from found
            return
    # Try all DAT children that might contain wavelength info
    for dat in substance.findall(".//DAT"):
        if dat.find(".//WAV") is not None or dat.find(".//WAVELENGTH") is not None:
            yield dat


def _get_text(el: ET.Element, path: str) -> str | None:
    """Get text from an XML element at the given path."""
    node = el.find(path)
    if node is not None and node.text:
        return node.text.strip()
    return None


def _parse_wavelength(text: str) -> list[float]:
    """Parse wavelength text which may contain ranges, lists, or single values."""
    if not text:
        return []
    # Remove units like "nm"
    text = re.sub(r"\s*nm\s*", "", text, flags=re.IGNORECASE)
    values = []
    # Split on comma or semicolon
    for part in re.split(r"[;,]", text):
        part = part.strip()
        if not part:
            continue
        # Range: "350-360" → take midpoint
        m = re.match(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", part)
        if m:
            values.append((float(m.group(1)) + float(m.group(2))) / 2)
            continue
        # Single number
        m = re.match(r"(\d+\.?\d*)", part)
        if m:
            values.append(float(m.group(1)))
    return values


def _parse_float(text: str) -> float | None:
    """Parse a float from text, returning None if unparseable."""
    if not text:
        return None
    m = re.search(r"(\d+\.?\d*(?:[eE][+-]?\d+)?)", text)
    return float(m.group(1)) if m else None


def _collect_all_tags(root: ET.Element, depth: int = 0, max_depth: int = 10) -> set[str]:
    """Recursively collect all XML tag names."""
    tags = set()
    if depth > max_depth:
        return tags
    tags.add(root.tag)
    for child in root:
        tags.update(_collect_all_tags(child, depth + 1, max_depth))
    return tags


# ---------------------------------------------------------------------------
# Progress tracking & incremental save
# ---------------------------------------------------------------------------
def load_progress() -> dict:
    """Load progress state from file."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"last_batch": -1, "total_records": 0, "total_batches": 0}


def save_progress(state: dict):
    """Save progress state."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(state, indent=2))


def append_raw_uv(records: list[dict]):
    """Append UV records to the raw CSV file."""
    RAW_UV_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RAW_UV_FILE.exists() and RAW_UV_FILE.stat().st_size > 0
    with open(RAW_UV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["xrn", "smiles", "wavelength_nm", "mec", "solvent"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------------------------
# Debug / test mode
# ---------------------------------------------------------------------------
def run_debug(session: ReaxysSession, xrn_list: list[str]):
    """Run a debug query on a few XRNs, dump raw XML, and print tag names."""
    test_xrns = xrn_list[:DEBUG_XRNS]
    print(f"\n=== DEBUG MODE: querying {len(test_xrns)} XRNs ===")
    print(f"  XRNs: {test_xrns}")

    session.connect()
    try:
        result_name, result_size = session.search(test_xrns)
        print(f"  Search returned {result_size} hits (result: {result_name})")

        if result_size == 0:
            print("  No hits found. Check that XRNs are valid.")
            return

        # Retrieve ALL fields (no field filter)
        raw_xml = session.retrieve_all_raw_xml(result_name, result_size, fields=None)

        # Save raw XML
        DEBUG_XML_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEBUG_XML_FILE.write_text(raw_xml, encoding="utf-8")
        print(f"  Raw XML saved to: {DEBUG_XML_FILE}")
        print(f"  XML size: {len(raw_xml):,} bytes")

        # Parse and list all tags
        root = ET.fromstring(raw_xml)
        tags = sorted(_collect_all_tags(root))
        print(f"\n  All XML tags found ({len(tags)}):")
        for t in tags:
            print(f"    - {t}")

        # Try extracting UV records with current field names
        records = extract_uv_records(root)
        print(f"\n  UV records extracted: {len(records)}")
        for rec in records[:10]:
            print(f"    {rec}")

        if not records:
            print(
                "\n  ⚠ No UV records extracted with current field names."
                "\n  Inspect the raw XML file to identify the correct tag names,"
                "\n  then update FIELDS_UV and the extract_uv_records() parser."
            )
    finally:
        session.disconnect()


# ---------------------------------------------------------------------------
# Full extraction
# ---------------------------------------------------------------------------
def run_full_extraction(session: ReaxysSession, xrn_list: list[str], batch_size: int = BATCH_SIZE):
    """Run full UV data extraction for all XRNs with progress tracking."""
    total = len(xrn_list)
    n_batches = (total + batch_size - 1) // batch_size
    progress = load_progress()
    start_batch = progress["last_batch"] + 1
    total_records = progress["total_records"]

    if start_batch > 0:
        print(f"\n  Resuming from batch {start_batch}/{n_batches} "
              f"({total_records} records so far)")
    else:
        print(f"\n  Starting extraction: {total} XRNs in {n_batches} batches")

    session.connect()
    try:
        for batch_idx in range(start_batch, n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total)
            batch_xrns = xrn_list[batch_start:batch_end]

            pct = (batch_idx + 1) / n_batches * 100
            print(
                f"\r  Batch {batch_idx + 1}/{n_batches} ({pct:.1f}%) "
                f"[XRN {batch_start+1}-{batch_end}] ...",
                end="",
                flush=True,
            )

            try:
                result_name, result_size = session.search(batch_xrns)

                if result_size > 0:
                    # Paginate through results
                    retrieved = 0
                    while retrieved < result_size:
                        first = retrieved + 1
                        last = min(retrieved + RETRIEVE_PAGE, result_size)
                        try:
                            root = session.retrieve(
                                result_name, first, last, fields=FIELDS_UV
                            )
                        except SessionExpiredError:
                            # Session expired mid-retrieval; re-search
                            result_name, result_size = session.search(batch_xrns)
                            root = session.retrieve(
                                result_name, first, last, fields=FIELDS_UV
                            )

                        records = extract_uv_records(root)
                        if records:
                            append_raw_uv(records)
                            total_records += len(records)
                        retrieved = last

                # Update progress
                progress["last_batch"] = batch_idx
                progress["total_records"] = total_records
                progress["total_batches"] = n_batches
                save_progress(progress)

            except Exception as e:
                print(f"\n  ERROR on batch {batch_idx}: {e}")
                print("  Saving progress and continuing...")
                progress["last_batch"] = batch_idx - 1  # retry this batch
                save_progress(progress)
                time.sleep(5)
                # Try to reconnect
                try:
                    session.connect()
                except Exception:
                    print("  Failed to reconnect. Exiting.")
                    break
                continue

            time.sleep(SLEEP_BETWEEN_BATCHES)

        print(f"\n\n  Extraction complete!")
        print(f"  Total UV records saved: {total_records}")
        print(f"  Output file: {RAW_UV_FILE}")

    finally:
        session.disconnect()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract UV-Vis data from Reaxys for Mamede et al. molecules"
    )
    parser.add_argument(
        "--test", action="store_true",
        help=f"Debug mode: query {DEBUG_XRNS} XRNs, dump raw XML, print tags",
    )
    parser.add_argument(
        "--apikey", type=str, default=None,
        help="Reaxys API caller name / key (overrides REAXYS_API_KEY env var)",
    )
    parser.add_argument(
        "--username", type=str, default="",
        help="Reaxys username (for named sessions; empty = anonymous/IP-based)",
    )
    parser.add_argument(
        "--password", type=str, default="",
        help="Reaxys password (for named sessions)",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to parsed Mamede CSV (default: auto-parse from supplementary file)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"XRNs per API batch (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Reset progress and start extraction from scratch",
    )
    args = parser.parse_args()

    # API key
    api_key = args.apikey or os.environ.get("REAXYS_API_KEY")
    if not api_key:
        print("WARNING: No API key provided. Using anonymous session (30 min timeout).")
        print("  Set REAXYS_API_KEY env var or use --apikey flag for longer sessions.")

    batch_size = args.batch_size

    # Parse or load Mamede data
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}")
            sys.exit(1)
        print(f"Loading parsed data from: {input_path}")
        records = load_parsed(input_path)
    elif PARSED_FILE.exists():
        print(f"Loading previously parsed data from: {PARSED_FILE}")
        records = load_parsed(PARSED_FILE)
    elif SUPP_FILE.exists():
        print(f"Parsing Mamede supplementary file: {SUPP_FILE}")
        records = parse_mamede_supplementary(SUPP_FILE)
        save_parsed(records, PARSED_FILE)
    else:
        print(f"ERROR: Mamede supplementary file not found at: {SUPP_FILE}")
        print("  Download 'Supplementary Information 2' from:")
        print("  https://www.nature.com/articles/s41598-021-03070-9")
        print(f"  Save it to: {SUPP_FILE}")
        sys.exit(1)

    xrn_list = [r["xrn"] for r in records if r.get("xrn")]
    print(f"  Total XRNs: {len(xrn_list)}")

    if not xrn_list:
        print("ERROR: No XRNs found in input data.")
        sys.exit(1)

    # Reset progress if requested
    if args.reset:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        if RAW_UV_FILE.exists():
            RAW_UV_FILE.unlink()
        print("  Progress reset.")

    # Create session and run
    session = ReaxysSession(
        api_key=api_key, username=args.username, password=args.password
    )

    if args.test:
        run_debug(session, xrn_list)
    else:
        run_full_extraction(session, xrn_list, batch_size=batch_size)


if __name__ == "__main__":
    main()
