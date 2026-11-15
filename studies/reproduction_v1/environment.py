#!/usr/bin/env python3
"""Explicit environment/snapshot sealing; imports no ML runtime by default.

--hardware runs bounded CUDA checks, never a language model. --download-models
explicitly downloads pinned licensed checkpoints. Archive only provenance,
never model weights, HF credentials or the HF cache directory.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
REVISION = re.compile(r'^[0-9a-f]{40}$')
NAME = re.compile(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')
RUNTIME_ENV = {
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'TOKENIZERS_PARALLELISM': 'false',
    'PYTHONHASHSEED': '0', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1', 'HF_HUB_ENABLE_HF_TRANSFER': '0',
}
TOP_LEVEL_EXPECTED = {
    'numpy': '2.2.6', 'scipy': '1.15.3', 'pandas': '2.2.3', 'PyYAML': '6.0.3',
    'torch': '2.8.0+cu128', 'transformers': '4.57.1', 'accelerate': '1.15.0',
    'huggingface_hub': '0.36.2', 'tokenizers': '0.22.2',
    'sentencepiece': '0.2.2', 'safetensors': '0.8.0',
}
TOKENIZER_FILES = {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json', 'vocab.json', 'vocab.txt', 'merges.txt',
                   'tokenizer.model', 'spiece.model'}
CONFIG_FILES = {'config.json', 'generation_config.json', 'model.safetensors.index.json'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Refusing symlink output')
    temp = path.with_name(path.name + '.partial')
    if temp.is_symlink():
        raise ValueError('Refusing symlink temporary output')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def load_config(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def gather_models(configs=()):
    """Collect pins without changing caller dtype/scientific settings."""
    configs = list(configs) or [PROJECT / 'studies/characterization_v1/frozen_run/study_config.json']
    found = {}
    def visit(value):
        if isinstance(value, dict):
            model = value.get('id', value.get('model_id'))
            revision = value.get('revision', value.get('model_revision'))
            if isinstance(model, str) and '/' in model and revision is not None:
                tokenizer = value.get('tokenizer_revision', revision)
                if not NAME.fullmatch(model) or not isinstance(revision, str) or not REVISION.fullmatch(revision):
                    raise ValueError('Model inventory requires a repository id and exact 40-hex commit')
                if tokenizer != revision:
                    raise ValueError('Current runtime requires identical model and tokenizer commits: ' + model)
                found[(model, revision)] = {'id': model, 'revision': revision, 'tokenizer_revision': tokenizer}
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    for config in configs: visit(load_config(config))
    if not found: raise ValueError('No pinned models found')
    return [found[key] for key in sorted(found)]


def selected_snapshot_names(names):
    selected = []
    for name in names:
        if not isinstance(name, str) or '/' in name or '\\' in name or name.startswith('.'):
            continue
        if name in TOKENIZER_FILES | CONFIG_FILES or re.fullmatch(r'model(?:-\d+-of-\d+)?\.safetensors', name):
            selected.append(name)
    if 'config.json' not in selected or not any(n.endswith('.safetensors') for n in selected):
        raise ValueError('Snapshot lacks config or root-level safetensors weights')
    if not any(n in selected for n in ('tokenizer.json', 'tokenizer.model', 'spiece.model', 'vocab.json', 'vocab.txt')):
        raise ValueError('Snapshot lacks tokenizer vocabulary')
    return sorted(set(selected))


def snapshot_inventory(snapshot, spec, expected_names=None):
    snapshot = Path(snapshot)
    revision = spec['revision']
    if not REVISION.fullmatch(revision) or snapshot.name != revision:
        raise ValueError('Snapshot directory is not the exact registered commit')
    if not snapshot.is_dir(): raise ValueError('Missing snapshot directory')
    # HF snapshots link into sibling blobs/: allow only this repository cache.
    repository_cache = snapshot.parent.parent.resolve()
    names = selected_snapshot_names([p.name for p in snapshot.iterdir() if p.is_file()])
    if expected_names is not None and names != sorted(expected_names):
        raise ValueError('Snapshot file inventory differs from requested remote inventory')
    files = {}
    if snapshot.is_symlink() or snapshot.parent.name != 'snapshots':
        raise ValueError('Use a standard Hub repository snapshot directory')
    for name in names:
        resolved = (snapshot / name).resolve(strict=True)
        if not resolved.is_relative_to(repository_cache) or not resolved.is_file():
            raise ValueError('Snapshot link escapes its repository cache')
        files[name] = {'bytes': resolved.stat().st_size, 'sha256': sha256(resolved)}
    config = json.loads((snapshot / 'config.json').read_text())
    index = snapshot / 'model.safetensors.index.json'
    weights = {n for n in names if n.endswith('.safetensors')}
    if index.exists():
        weight_map = json.loads(index.read_text()).get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map: raise ValueError('Malformed safetensors index')
        if set(weight_map.values()) != weights: raise ValueError('Index shard coverage differs from sealed weight files')
    elif weights != {'model.safetensors'}:
        raise ValueError('Sharded safetensors require a complete index')
    return {'schema_version': 1, **spec, 'files': files,
            'total_bytes': sum(v['bytes'] for v in files.values()),
            'model_type': config.get('model_type'), 'hidden_size': config.get('hidden_size'),
            'num_hidden_layers': config.get('num_hidden_layers'),
            'tokenizer_file_names': sorted(set(names) & TOKENIZER_FILES),
            'credentials_included': False, 'weights_included_in_result_archive': False}


def model_seal_name(spec):
    return spec['id'].replace('/', '__') + '__' + spec['revision'] + '.json'


def seal_models(models, output, download=False, cache_dir=None):
    from huggingface_hub import HfApi, snapshot_download
    directory = Path(output) / 'model_inventory'
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for spec in models:
        print(('DOWNLOAD_OR_VERIFY ' if download else 'VERIFY_LOCAL ') + spec['id'] + '@' + spec['revision'], flush=True)
        seal_path = directory / model_seal_name(spec)
        previous = json.loads(seal_path.read_text()) if seal_path.exists() else None
        if download:
            # Normal HF cache/environment provides authentication; never log it.
            info = HfApi().model_info(spec['id'], revision=spec['revision'], files_metadata=False)
            if info.sha != spec['revision']: raise ValueError('Hub resolved a different commit')
            names = selected_snapshot_names([s.rfilename for s in info.siblings])
            snapshot = snapshot_download(repo_id=spec['id'], revision=spec['revision'],
                                         allow_patterns=names, cache_dir=cache_dir, max_workers=2)
        else:
            if previous is None: raise ValueError('Local verification needs an existing seal; run explicit download/seal first')
            names = sorted(previous['files'])
            snapshot = snapshot_download(repo_id=spec['id'], revision=spec['revision'],
                                         allow_patterns=names, cache_dir=cache_dir, local_files_only=True)
        current = snapshot_inventory(snapshot, spec, names)
        if previous is not None and current != previous:
            raise ValueError('Pinned model/tokenizer files changed; preserve existing seal')
        if previous is None: atomic_json(seal_path, current)
        result.append({'id': spec['id'], 'revision': spec['revision'],
                       'seal_file': 'model_inventory/' + seal_path.name,
                       'seal_sha256': sha256(seal_path), 'total_bytes': current['total_bytes']})
    atomic_json(Path(output) / 'model_inventory.json', {'models': result, 'credentials_included': False})
    return result


def pip_report_to_lock(report, destination):
    data = json.loads(Path(report).read_text())
    installs = data.get('install')
    if not isinstance(installs, list) or not installs: raise ValueError('Pip report has no resolved wheel closure')
    lines, seen = [], set()
    for item in installs:
        meta = item['metadata']; name, version = meta['name'], meta['version']
        canonical = re.sub(r'[-_.]+', '-', name).lower()
        if canonical in seen: raise ValueError('Duplicate package in dependency closure')
        seen.add(canonical)
        download = item['download_info']; url = download['url']; parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or not parsed.path.endswith('.whl'):
            raise ValueError('Lock accepts credential-free HTTPS wheel URLs only')
        digest = download.get('archive_info', {}).get('hashes', {}).get('sha256')
        if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('Every resolved wheel needs a SHA256 hash')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or not isinstance(version, str) or '\n' in version or '\r' in version:
            raise ValueError('Malformed package metadata')
        lines.append((canonical, f'# {name}=={version}\n{name} @ {url} --hash=sha256:{digest}'))
    if not {'torch', 'numpy', 'scipy', 'transformers', 'huggingface-hub', 'tokenizers'} <= seen:
        raise ValueError('Pip report is missing required scientific packages')
    text = '# Complete wheel closure resolved before installation. Platform/Python-specific.\n' + '\n'.join(v for _, v in sorted(lines)) + '\n'
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or (destination.exists() and destination.read_text() != text):
        raise ValueError('Refusing to replace a different dependency lock')
    destination.write_text(text, encoding='utf-8')
    return {'packages': len(seen), 'sha256': sha256(destination)}


def available_ram_bytes():
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith('MemAvailable:'): return int(line.split()[1]) * 1024
    return None


def environment_snapshot(output, hardware=False, minimum_free_gib=0, minimum_ram_gib=0,
                         minimum_vram_gib=44, require_versions=True):
    if any(not isinstance(v, (int, float)) or v < 0 or v != v or v == float('inf')
           for v in (minimum_free_gib, minimum_ram_gib, minimum_vram_gib)):
        raise ValueError('Resource minimums must be finite nonnegative numbers')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in TOP_LEVEL_EXPECTED:
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = None
    free, ram = shutil.disk_usage(output).free, available_ram_bytes()
    facts = {'schema_version': 1, 'python': platform.python_version(),
             'platform_system': platform.system(), 'machine': platform.machine(),
             'packages': packages, 'cuda_checked': False,
             'free_disk_bytes': free, 'available_host_ram_bytes': ram,
             'deterministic_environment': {k: os.environ.get(k) for k in RUNTIME_ENV},
             'credentials_included': False, 'models_downloaded': False,
             'cross_hardware_bitwise_guarantee': False,
             'required_disk_reserve_bytes': int(minimum_free_gib * 2**30)}
    issues = []
    if free < minimum_free_gib * 2**30: issues.append('Insufficient free disk for the requested reserve')
    if minimum_ram_gib and (ram is None or ram < minimum_ram_gib * 2**30):
        issues.append('Available RAM below requested minimum or unavailable')
    if hardware:
        if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 11):
            issues.append('Registered GPU environment requires Linux x86_64 Python 3.11')
        if require_versions:
            issues.extend('Dependency differs: ' + k for k, v in TOP_LEVEL_EXPECTED.items() if packages[k] != v)
        for key, value in RUNTIME_ENV.items():
            if key == 'PYTHONHASHSEED' and os.environ.get(key) != value:
                issues.append('PYTHONHASHSEED=0 must be set before launching Python')
            elif key != 'PYTHONHASHSEED': os.environ[key] = value
        if not issues:
            import torch
            if not torch.cuda.is_available(): issues.append('CUDA is unavailable')
            else:
                torch.use_deterministic_algorithms(True)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.manual_seed(0)
                gpu = torch.cuda.get_device_properties(0)
                facts['gpu'] = {'name': gpu.name, 'total_memory_bytes': gpu.total_memory,
                                'compute_capability': list(torch.cuda.get_device_capability(0)),
                                'device_count': torch.cuda.device_count()}
                facts['torch_cuda_build'] = torch.version.cuda
                facts['cudnn_version'] = torch.backends.cudnn.version()
                facts['cuda_checked'] = True
                if gpu.total_memory < minimum_vram_gib * 2**30: issues.append('GPU memory below requested minimum')
                if not torch.cuda.is_bf16_supported(): issues.append('GPU does not support BF16')
                x = torch.arange(256, device='cuda', dtype=torch.float32).reshape(16, 16).to(torch.bfloat16)
                a, b = x @ x.T, x @ x.T
                if not torch.equal(a, b) or not bool(torch.isfinite(a).all()):
                    issues.append('Deterministic finite BF16 kernel check failed')
                del x, a, b
                try:
                    driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True, timeout=10)
                    facts['nvidia_driver_versions'] = sorted(set(driver.strip().splitlines()))
                except (OSError, subprocess.SubprocessError): facts['nvidia_driver_versions'] = None
        facts['deterministic_environment'] = {k: os.environ.get(k) for k in RUNTIME_ENV}
    if hardware and not issues:
        stable = {key: facts.get(key) for key in ('python', 'platform_system', 'machine', 'packages',
                  'gpu', 'torch_cuda_build', 'cudnn_version', 'nvidia_driver_versions', 'deterministic_environment')}
        identity_path = output / 'runtime_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != stable:
            issues.append('Hardware/runtime identity changed; preserve this run and use a fresh provenance directory')
        elif not identity_path.exists():
            atomic_json(identity_path, stable)
    facts['status'] = 'failed' if issues else ('hardware_passed' if hardware else 'source_inspection_only')
    facts['issues'] = issues
    atomic_json(output / ('hardware_preflight.json' if hardware else 'environment_inspection.json'), facts)
    if issues: raise ValueError('; '.join(issues))
    return facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, action='append', default=[])
    parser.add_argument('--download-models', action='store_true')
    parser.add_argument('--verify-models', action='store_true')
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--minimum-free-gib', type=float, default=0)
    parser.add_argument('--minimum-ram-gib', type=float, default=0)
    parser.add_argument('--minimum-vram-gib', type=float, default=44)
    parser.add_argument('--pip-report', type=Path)
    parser.add_argument('--lock-output', type=Path)
    args = parser.parse_args(argv)
    if args.pip_report or args.lock_output:
        if not args.pip_report or not args.lock_output: parser.error('Supply both --pip-report and --lock-output')
        print(json.dumps(pip_report_to_lock(args.pip_report, args.lock_output), indent=2)); return
    if args.output is None: parser.error('--output is required')
    if args.output.is_symlink(): parser.error('Output directory cannot be a symlink')
    models = gather_models(args.config)
    plan = {'models': models, 'execution_dtype_from_each_scientific_config': True,
            'historical_float16_not_changed': True, 'current_bfloat16_not_changed': True}
    plan_path = args.output / 'model_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Model panel changed; use a fresh provenance output')
    if not plan_path.exists(): atomic_json(plan_path, plan)
    facts = environment_snapshot(args.output, args.hardware, args.minimum_free_gib, args.minimum_ram_gib, args.minimum_vram_gib)
    if args.download_models or args.verify_models:
        seal_models(models, args.output, download=args.download_models, cache_dir=args.cache_dir)
    print(json.dumps({'status': facts['status'], 'pinned_models': len(models),
                      'download_requested': args.download_models, 'model_execution': False}, indent=2))


if __name__ == '__main__':
    try: main()
    except (ValueError, OSError) as error:
        print('environment: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
