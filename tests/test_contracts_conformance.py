"""ImageHarbor against the shared photo-contracts package.

photo-contracts (github.com/conradstorz/photo-contracts) is the single written
definition of what ImageHarbor and its sibling tools agree on: digest
spelling, the date-tree layout, the filename grammar, and the --json report
shapes. If ImageHarbor drifts from it -- or the contract changes under
ImageHarbor -- this file is where that shows up.

Dev-only dependency: nothing under imageharbor/ imports photo_contracts.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("photo_contracts")
jsonschema = pytest.importorskip("jsonschema")

from click.testing import CliRunner  # noqa: E402
from photo_contracts import contract, digest, layout  # noqa: E402
from PIL import Image  # noqa: E402

import imageharbor  # noqa: E402
from imageharbor import hashing  # noqa: E402
from imageharbor.cli import main  # noqa: E402
from imageharbor.takeout import index_reader  # noqa: E402


def _check(doc: dict, schema_name: str) -> None:
    jsonschema.validate(doc, contract.load_schema(schema_name))


@pytest.fixture()
def library(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(src / "holiday card.jpg")
    Image.new("RGB", (8, 8), (10, 200, 10)).save(src / "IMG_0001.jpg")
    (src / "IMG_0001 copy.jpg").write_bytes((src / "IMG_0001.jpg").read_bytes())
    return src, dest


def test_process_enrich_verify_reports_match_schemas(library):
    src, dest = library
    p = imageharbor.process(src, dest)
    _check(p.to_dict(), "process_report")
    assert p.counts[contract.COPIED] == 2 and p.counts[contract.DUPLICATE] == 1

    e = imageharbor.enrich(dest)
    _check(e.to_dict(), "enrich_report")

    v = imageharbor.verify(dest)
    _check(v.to_dict(), "verify_report")
    assert v.ok


def test_cli_json_documents_match_schemas_and_exit_codes(library):
    src, dest = library
    runner = CliRunner()
    r = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest), "--json"])
    assert r.exit_code == contract.EXIT_OK, r.output
    _check(json.loads(r.stdout), "process_report")

    r = runner.invoke(main, ["verify", str(dest), "--json"])
    assert r.exit_code == contract.EXIT_OK, r.output
    _check(json.loads(r.stdout), "verify_report")

    r = runner.invoke(main, ["verify", str(dest / "nope"), "--json"])
    assert r.exit_code == contract.EXIT_CONFIG_OR_ABORT


def test_organized_tree_obeys_layout_and_digest_contract(library):
    src, dest = library
    # A source with a known capture date (from its filename) so the dated
    # YYYY/YYYY-MM/ branch is exercised, not only Undated/.
    Image.new("RGB", (8, 8), (10, 10, 200)).save(src / "2019-07-04 beach.jpg")
    imageharbor.process(src, dest)
    seen = 0
    dated = 0
    for f in dest.rglob("*.jpg"):
        parsed = layout.parse_filename(f.name)
        assert parsed is not None, f
        rel = f.parent.relative_to(dest).as_posix()
        assert layout.folder_matches_name(rel, parsed), (rel, f.name)
        assert digest.sha256_b64url(f) == parsed.digest
        if f.name.startswith("2019-07-04"):
            # the known date must survive into both folder and filename
            assert rel == "2019/2019-07", rel
            dated += 1
        else:
            assert rel == "Undated", rel
        seen += 1
    assert seen == 3 and dated == 1


def test_digest_encoding_and_filename_parsing_agree(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"photo-contracts")
    d = hashing.compute_sha256_b64url(f)
    assert d == digest.sha256_b64url(f)
    # same stem, both parsers (ImageHarbor keeps two copies of this logic)
    stem = f"2019-07-04-beach_{d}"
    assert hashing.extract_digest_from_stem(stem) == layout.parse_filename(stem + ".jpg").digest


def test_takeout_index_version_matches_contract():
    assert index_reader.SCHEMA_VERSION == max(contract.TAKEOUT_INDEX_SUPPORTED_VERSIONS)
