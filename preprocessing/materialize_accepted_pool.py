"""Materialize the synthetic images accepted by lesion-calibrated thresholds."""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / 'data' / 'eval'
SYNTH_ROOT = REPO_ROOT / 'data' / 'synthetic'
DETAIL_CSV = EVAL_DIR / 'memorization_ddpm64_detail.csv'
THRESHOLDS_JSON = EVAL_DIR / 'memorization_thresholds_lesion.json'
SPLITS_CSV = REPO_ROOT / 'data' / 'splits.csv'
SYNTHETIC_DIR = SYNTH_ROOT / 'ddpm64'
DESTINATION = SYNTH_ROOT / 'ddpm64_accepted'
REQUIRED_COLUMNS = {'class', 'file', 'lpips_nn', 'inception_nn'}


def _safe_component(value, label):
    """Return a path component after rejecting ambiguity and traversal."""
    if not isinstance(value, str) or value != value.strip() or not value:
        raise RuntimeError(f'invalid {label}: {value!r}')
    if value in ('.', '..') or '/' in value or '\\' in value:
        raise RuntimeError(f'invalid {label}: {value!r}')
    return value


def _finite_float(value, label):
    """Parse one finite numeric field."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f'invalid {label}: {value!r}') from error
    if not math.isfinite(result):
        raise RuntimeError(f'invalid {label}: {value!r}')
    return result


def _read_detail(path):
    """Load and validate the memorization detail rows."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f'memorization detail CSV not found: {path}')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise RuntimeError(
                f'memorization detail CSV is missing columns: '
                f'{sorted(missing)}')
        rows = []
        seen = {}
        for line, raw in enumerate(reader, start=2):
            cls = _safe_component(raw['class'], f'class on line {line}')
            name = _safe_component(raw['file'], f'file on line {line}')
            if Path(name).suffix.lower() != '.png':
                raise RuntimeError(
                    f'detail line {line} is not a PNG: {name!r}')
            key = (cls, name)
            folded = (cls.casefold(), name.casefold())
            if folded in seen:
                raise RuntimeError(
                    f'duplicate detail image on lines {seen[folded]} and '
                    f'{line}: {cls}/{name}')
            seen[folded] = line
            rows.append({
                'class': cls,
                'file': name,
                'lpips_nn': _finite_float(
                    raw['lpips_nn'], f'lpips_nn on line {line}'),
                'inception_nn': _finite_float(
                    raw['inception_nn'], f'inception_nn on line {line}'),
                'key': key,
            })
    if not rows:
        raise RuntimeError('memorization detail CSV has no rows')
    return sorted(rows, key=lambda row: row['key'])


def _read_thresholds(path, classes):
    """Load the lesion-calibrated thresholds needed by the detail rows."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f'lesion threshold JSON not found: {path}')
    try:
        with path.open(encoding='utf-8') as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f'cannot read lesion threshold JSON: {path}') \
            from error
    if not isinstance(payload, dict) or not isinstance(
            payload.get('thresholds'), dict):
        raise RuntimeError('lesion threshold JSON has no thresholds object')

    thresholds = {}
    for cls in sorted(classes):
        values = payload['thresholds'].get(cls)
        if not isinstance(values, dict):
            raise RuntimeError(f'lesion thresholds are missing class {cls!r}')
        if 'lpips' not in values or 'inception' not in values:
            raise RuntimeError(
                f'lesion thresholds for {cls!r} require lpips and inception')
        thresholds[cls] = {
            'lpips': _finite_float(values['lpips'], f'{cls} LPIPS threshold'),
            'inception': _finite_float(
                values['inception'], f'{cls} Inception threshold'),
        }
    return payload, thresholds


def _validate_split_provenance(payload, splits_csv):
    """Validate a stored split hash when both the hash and split file exist."""
    stored = payload.get('splits_sha1')
    if splits_csv is None or stored is None:
        return None
    if not isinstance(stored, str):
        raise RuntimeError('splits_sha1 in lesion threshold JSON is not text')
    path = Path(splits_csv)
    if not path.is_file():
        raise RuntimeError(f'splits CSV not found for provenance check: {path}')
    actual = hashlib.sha1(path.read_bytes()).hexdigest()
    if stored.lower() != actual:
        raise RuntimeError(
            'lesion thresholds were built from a different splits.csv: '
            f'expected {stored}, found {actual}')
    return actual


def _source_inventory(root, label, required, expected_classes):
    """Return the exact class/name inventory for one flat class tree."""
    root = Path(root)
    if root.is_symlink():
        raise RuntimeError(f'{label} source tree may not be a symlink: {root}')
    if not root.exists():
        if required:
            raise RuntimeError(f'{label} source tree not found: {root}')
        return {}
    if not root.is_dir():
        raise RuntimeError(f'{label} source tree is not a directory: {root}')

    inventory = {}
    for class_path in sorted(root.iterdir(), key=lambda path: path.name):
        if class_path.is_symlink() or not class_path.is_dir():
            raise RuntimeError(f'unexpected source entry: {class_path}')
        cls = _safe_component(class_path.name, f'{label} class directory')
        if cls not in expected_classes:
            raise RuntimeError(
                f'unexpected source class directory: {class_path}')
        for image_path in sorted(class_path.iterdir(),
                                 key=lambda path: path.name):
            if image_path.is_symlink() or not image_path.is_file():
                raise RuntimeError(f'unexpected source entry: {image_path}')
            if image_path.suffix.lower() != '.png':
                raise RuntimeError(f'unexpected source file: {image_path}')
            name = _safe_component(image_path.name, f'{label} image file')
            key = (cls, name)
            folded = (cls.casefold(), name.casefold())
            if folded in inventory:
                other = inventory[folded][1]
                raise RuntimeError(
                    f'duplicate source image: {other} and {image_path}')
            inventory[folded] = (key, image_path)
    return inventory


def _is_within(path, directory):
    """Whether path is equal to or below directory."""
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _validate_destination(destination, source_roots, input_files):
    """Resolve a destination that cannot overlap any source or input."""
    raw = Path(destination).expanduser().absolute()
    if raw.is_symlink():
        raise RuntimeError(f'destination may not be a symlink: {raw}')
    destination = raw.resolve()
    if destination == destination.parent:
        raise RuntimeError('destination may not be a filesystem root')

    for root in source_roots:
        root = Path(root).expanduser().resolve()
        if _is_within(destination, root) or _is_within(root, destination):
            raise RuntimeError(
                f'destination overlaps source tree: {destination} and {root}')
    for path in input_files:
        if path is None:
            continue
        path = Path(path).expanduser().resolve()
        if _is_within(path, destination):
            raise RuntimeError(
                f'destination contains an input artifact: {path}')
    if os.path.lexists(destination) and not destination.is_dir():
        raise RuntimeError(
            f'destination exists and is not a directory: {destination}')
    return destination


def _replace_tree(staged, destination):
    """Install a staged sibling, restoring the old tree if install fails."""
    if not os.path.lexists(destination):
        os.replace(staged, destination)
        return

    backup_container = Path(tempfile.mkdtemp(
        prefix=f'.{destination.name}.backup-', dir=destination.parent))
    backup = backup_container / 'previous'
    try:
        os.replace(destination, backup)
    except BaseException:
        shutil.rmtree(backup_container)
        raise
    try:
        os.replace(staged, destination)
    except BaseException as install_error:
        try:
            os.replace(backup, destination)
        except BaseException as rollback_error:
            raise RuntimeError(
                f'install failed and previous destination is at {backup}') \
                from rollback_error
        shutil.rmtree(backup_container)
        raise install_error
    shutil.rmtree(backup_container)


def materialize_accepted_pool(detail_csv, thresholds_json, synthetic_dir,
                              destination, splits_csv=None,
                              flagged_dir=None):
    """Build and atomically install the exact lesion-accepted image tree."""
    synthetic_dir = Path(synthetic_dir).expanduser().absolute()
    if flagged_dir is None:
        flagged_dir = Path(str(synthetic_dir) + '_flagged')
    else:
        flagged_dir = Path(flagged_dir).expanduser().absolute()
    if synthetic_dir.resolve() == flagged_dir.resolve():
        raise RuntimeError('active and flagged source trees must differ')

    rows = _read_detail(detail_csv)
    classes = {row['class'] for row in rows}
    payload, thresholds = _read_thresholds(thresholds_json, classes)
    split_sha1 = _validate_split_provenance(payload, splits_csv)
    destination = _validate_destination(
        destination, (synthetic_dir, flagged_dir),
        (detail_csv, thresholds_json, splits_csv))

    active = _source_inventory(
        synthetic_dir, 'active', required=True, expected_classes=classes)
    # only populated when memorization.py ran with --quarantine
    flagged = _source_inventory(
        flagged_dir, 'flagged', required=False, expected_classes=classes)
    sources = dict(active)
    duplicates = []
    for folded, value in flagged.items():
        if folded in sources:
            duplicates.append((sources[folded][1], value[1]))
        else:
            sources[folded] = value
    if duplicates:
        first, second = duplicates[0]
        raise RuntimeError(f'duplicate source image: {first} and {second}')

    expected = {row['key'] for row in rows}
    actual = {value[0] for value in sources.values()}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    problems = []
    if missing:
        problems.append(
            f'missing source images: {missing[:3]}'
            f'{" ..." if len(missing) > 3 else ""}')
    if unexpected:
        problems.append(
            f'unexpected source images: {unexpected[:3]}'
            f'{" ..." if len(unexpected) > 3 else ""}')
    if problems:
        raise RuntimeError('; '.join(problems))

    accepted = []
    for row in rows:
        threshold = thresholds[row['class']]
        flagged_by_lesion_threshold = (
            row['lpips_nn'] < threshold['lpips'] or
            row['inception_nn'] < threshold['inception'])
        if not flagged_by_lesion_threshold:
            folded = (row['class'].casefold(), row['file'].casefold())
            accepted.append((row['key'], sources[folded][1]))

    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(
        prefix=f'.{destination.name}.tmp-', dir=destination.parent))
    try:
        for cls in sorted(classes):
            (staged / cls).mkdir()
        for (cls, name), source in accepted:
            shutil.copyfile(source, staged / cls / name)
        _replace_tree(staged, destination)
    finally:
        if staged.exists():
            shutil.rmtree(staged)

    return {
        'total': len(rows),
        'accepted': len(accepted),
        'rejected': len(rows) - len(accepted),
        'classes': len(classes),
        'splits_sha1': split_sha1,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Materialize the lesion-calibrated accepted pool.')
    parser.add_argument('--detail-csv', default=DETAIL_CSV)
    parser.add_argument('--thresholds-json', default=THRESHOLDS_JSON)
    parser.add_argument('--synthetic-dir', default=SYNTHETIC_DIR)
    parser.add_argument(
        '--flagged-dir',
        help='quarantine tree; defaults to <synthetic-dir>_flagged')
    parser.add_argument('--splits-csv', default=SPLITS_CSV)
    parser.add_argument('--destination', default=DESTINATION)
    args = parser.parse_args()

    result = materialize_accepted_pool(
        args.detail_csv, args.thresholds_json, args.synthetic_dir,
        args.destination, args.splits_csv, args.flagged_dir)
    provenance = result['splits_sha1'] or 'unavailable'
    print(
        f"wrote {args.destination}: {result['accepted']} accepted, "
        f"{result['rejected']} rejected across {result['classes']} classes "
        f"(splits SHA-1 {provenance})")


if __name__ == '__main__':
    main()
