#!/usr/bin/env python3
"""Build/run identical Release benchmarks and compare validated CPU/CUDA timings.

Uses only Python's standard library. Each backend lives in a separate process
and build directory, avoiding Thrust backend/ABI conflicts in one executable.
"""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import platform
import subprocess


def run(command, log, env=None):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    with log.open('w') as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                       check=True, env=env)


def probe(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return str(error)


def main():
    source = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-backend', choices=['CPP', 'OMP', 'TBB'], default='CPP')
    parser.add_argument('--architecture', default='native', help='e.g. 89 for compute capability 8.9')
    parser.add_argument('--sizes', default='1000,10000,100000')
    parser.add_argument('--entries-per-row', type=int, default=9)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--layout', choices=['csr', 'csc', 'both'], default='both')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--threads', type=int, help='Explicit OpenMP thread count; does not limit TBB')
    parser.add_argument('--cccl-root', type=Path)
    parser.add_argument('--build-root', type=Path, default=source / 'build-benchmark')
    parser.add_argument('--output', type=Path, default=source / 'benchmark-results')
    args = parser.parse_args()
    for key in ['entries_per_row', 'iterations', 'samples', 'warmup', 'jobs', 'threads']:
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(key.replace('_', '-') + ' must be positive')
    try:
        sizes = [int(n) for n in args.sizes.split(',')]
        if not sizes or min(sizes) < 1 or len(set(sizes)) != len(sizes):
            raise ValueError()
    except ValueError:
        parser.error('--sizes must be distinct positive comma-separated integers')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # A failed new run must not leave a stale successful comparison behind.
    for name in ['report.md', 'comparison.csv']:
        (args.output / name).unlink(missing_ok=True)
    env = os.environ.copy()
    if args.threads:
        env['OMP_NUM_THREADS'] = str(args.threads)
    cpu_name = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu_name = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                         if line.startswith('model name')), cpu_name)
    metadata = {
        'timestamp': datetime.datetime.now().astimezone().isoformat(),
        'platform': platform.platform(), 'cpu': cpu_name,
        'gpu': probe(['nvidia-smi', '--query-gpu=name,driver_version,compute_cap', '--format=csv,noheader']),
        'cmake': probe(['cmake', '--version']),
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'thread_environment': {k: env.get(k) for k in ['OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES']},
        'status': 'running',
    }
    metadata_path = args.output / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    try:
        for backend in [args.cpu_backend, 'CUDA']:
            build = args.build_root.resolve() / backend.lower()
            command = ['cmake', '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                       '-DAPSC_BUILD_BENCHMARKS=ON', '-DAPSC_BUILD_TESTS=OFF', '-DAPSC_BUILD_EXAMPLES=OFF',
                       '-DAPSC_THRUST_BACKEND=' + backend]
            if backend == 'CUDA':
                command.append('-DCMAKE_CUDA_ARCHITECTURES=' + args.architecture)
            if args.cccl_root:
                command.append('-DCCCL_ROOT=' + str(args.cccl_root.resolve()))
            run(command, args.output / (backend.lower() + '-configure.log'), env)
            run(['cmake', '--build', build, '--config', 'Release', '--target', 'sparse_benchmark',
                 '--parallel', str(args.jobs)], args.output / (backend.lower() + '-build.log'), env)
            name = 'sparse_benchmark.exe' if os.name == 'nt' else 'sparse_benchmark'
            executable = build / name
            if not executable.exists():
                executable = build / 'Release' / name
            run([executable, '--sizes', args.sizes, '--entries-per-row', str(args.entries_per_row),
                 '--iterations', str(args.iterations), '--samples', str(args.samples),
                 '--warmup', str(args.warmup), '--layout', args.layout,
                 '--csv', args.output / (backend.lower() + '.csv')],
                args.output / (backend.lower() + '-run.log'), env)
        def read(backend):
            with (args.output / (backend.lower() + '.csv')).open() as stream:
                rows = list(csv.DictReader(stream))
            keyed = {(r['layout'], r['rows'], r['nnz'], r['operation']): r for r in rows}
            if len(keyed) != len(rows) or not rows:
                raise RuntimeError('Empty/duplicate benchmark results')
            return keyed
        cpu, gpu = read(args.cpu_backend), read('CUDA')
        if cpu.keys() != gpu.keys():
            raise RuntimeError('CPU and GPU workloads differ')
        comparison = []
        for key, host in cpu.items():
            device = gpu[key]
            if host['build'] != 'Release' or device['build'] != 'Release':
                raise RuntimeError('Both builds must be Release')
            if not math.isclose(float(host['checksum']), float(device['checksum']), rel_tol=1e-8, abs_tol=1e-8):
                raise RuntimeError('CPU/GPU checksum mismatch: ' + str(key))
            host_ms, device_ms = float(host['median_ms']), float(device['median_ms'])
            if not (math.isfinite(host_ms) and math.isfinite(device_ms) and host_ms > 0 and device_ms > 0):
                raise RuntimeError('Invalid timing: ' + str(key))
            comparison.append(dict(layout=key[0], rows=key[1], nnz=key[2], operation=key[3],
                                   cpu_backend=args.cpu_backend, cpu_ms=host_ms, gpu_ms=device_ms,
                                   speedup_cpu_over_gpu=host_ms / device_ms))
        with (args.output / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
            writer.writeheader(); writer.writerows(comparison)
        report = ['# CPU / GPU benchmark results', '',
                  f'CPU: {cpu_name}; backend: {args.cpu_backend}.', '',
                  f'GPU: {metadata["gpu"]}.', '',
                  'Release builds; double precision; medians in milliseconds per call. '
                  'Speedup = CPU / GPU (>1 favors GPU). All correctness checks passed.', '',
                  '| Layout | Rows | Operation | CPU ms | GPU ms | Speedup |',
                  '| --- | ---: | --- | ---: | ---: | ---: |']
        for row in comparison:
            report.append(f'| {row["layout"]} | {row["rows"]} | {row["operation"]} | '
                          f'{row["cpu_ms"]:.6f} | {row["gpu_ms"]:.6f} | {row["speedup_cpu_over_gpu"]:.2f}x |')
        report += ['', 'See metadata.json for settings and environment; backend CSV files contain '
                   'sample ranges and separately measured setup costs. These results describe this '
                   'workload and machine, not a universal GPU speedup.', '']
        (args.output / 'report.md').write_text('\n'.join(report))
        metadata['status'] = 'passed'
        print('\n'.join(report), flush=True)
    except Exception as error:
        metadata['status'] = 'failed'
        metadata['error'] = str(error)
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
