#!/usr/bin/env python3
"""safetensors alignment / coalescing analyzer.

Decides whether an O_DIRECT io_uring reader is viable, and how much read
amplification alignment padding costs.

safetensors layout:
    [0..8)     u64 LE header size N
    [8..8+N)   JSON header
    [8+N..EOF) tensor byte buffer
Tensor `data_offsets` are relative to the byte buffer, so the absolute file
offset of a tensor is 8 + N + start.

O_DIRECT requires offset and length to be block (typically 4096) aligned.
Usage: st_align.py <model_dir_or_files...> [--block 4096]
"""
import glob
import json
import os
import struct
import sys
from collections import Counter

DTYPE_SIZE = {
    'F64': 8, 'I64': 8, 'F32': 4, 'I32': 4, 'BF16': 2, 'F16': 2, 'I16': 2,
    'I8': 1, 'U8': 1, 'BOOL': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
}


def parse_shard(path):
    with open(path, 'rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError('truncated header length')
        n = struct.unpack('<Q', raw)[0]
        if n == 0 or n > 512 * 1024 * 1024:
            raise ValueError(f'implausible header size {n}')
        head = f.read(n)
        if len(head) != n:
            raise ValueError('truncated JSON header')
    hdr = json.loads(head.decode('utf-8'))
    meta = hdr.pop('__metadata__', None)
    return n, hdr, meta


def ceil_to(x, b):
    return -(-x // b) * b


def main(paths, block):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, '*.safetensors'))))
        else:
            files.append(p)
    if not files:
        sys.exit('no .safetensors files found')

    grand_tensor_bytes = 0
    grand_read_bytes = 0
    grand_tensors = 0
    grand_requests = 0
    grand_unaligned_off = 0
    grand_unaligned_size = 0
    dtypes = Counter()

    print(f'block size = {block}\n')
    for fp in files:
        try:
            n, hdr, meta = parse_shard(fp)
        except Exception as e:                                  # noqa: BLE001
            print(f'{os.path.basename(fp)}: FAILED {e}')
            continue

        base = 8 + n
        fsize = os.path.getsize(fp)
        buf_size = fsize - base

        items = []
        for name, info in hdr.items():
            s, e = info['data_offsets']
            items.append((s, e, name, info['dtype'], tuple(info['shape'])))
        items.sort()

        tb = sum(e - s for s, e, *_ in items)
        unal_off = sum(1 for s, e, *_ in items if (base + s) % block)
        unal_sz = sum(1 for s, e, *_ in items if (e - s) % block)
        for _, _, _, dt, _ in items:
            dtypes[dt] += 1

        # gaps between consecutive tensors inside the buffer
        gaps = [items[i][0] - items[i - 1][1] for i in range(1, len(items))]
        gap_bytes = sum(gaps)
        head_gap = items[0][0] if items else 0
        tail_gap = buf_size - items[-1][1] if items else 0

        # coalesce into block-aligned read ranges
        ranges = []
        for s, e, *_ in items:
            a0 = (base + s) // block * block
            a1 = ceil_to(base + e, block)
            if ranges and a0 <= ranges[-1][1]:
                ranges[-1][1] = max(ranges[-1][1], a1)
            else:
                ranges.append([a0, a1])
        read_bytes = sum(b - a for a, b in ranges)

        grand_tensor_bytes += tb
        grand_read_bytes += read_bytes
        grand_tensors += len(items)
        grand_requests += len(ranges)
        grand_unaligned_off += unal_off
        grand_unaligned_size += unal_sz

        amp = read_bytes / tb if tb else 0.0
        print(f'== {os.path.basename(fp)}')
        print(f'   header={n}B  file={fsize/1024**3:.3f} GiB  buffer={buf_size/1024**3:.3f} GiB  '
              f'tensors={len(items)}')
        print(f'   tensor bytes={tb/1024**3:.3f} GiB   '
              f'internal gaps={gap_bytes}B  head={head_gap}B  tail={tail_gap}B')
        print(f'   NOT aligned to {block}: offset {unal_off}/{len(items)}   size {unal_sz}/{len(items)}')
        print(f'   requests: naive={len(items)}  coalesced={len(ranges)}  '
              f'read={read_bytes/1024**3:.3f} GiB  amplification={amp:.4f}x')
        print()

    if grand_tensor_bytes:
        print('#### TOTAL')
        print(f'   shards={len(files)} tensors={grand_tensors}')
        print(f'   tensor bytes={grand_tensor_bytes/1024**3:.3f} GiB')
        print(f'   read bytes after alignment={grand_read_bytes/1024**3:.3f} GiB')
        print(f'   amplification={grand_read_bytes/grand_tensor_bytes:.4f}x')
        print(f'   unaligned offset: {grand_unaligned_off}/{grand_tensors} '
              f'({100*grand_unaligned_off/max(1,grand_tensors):.1f}%)')
        print(f'   unaligned size:   {grand_unaligned_size}/{grand_tensors} '
              f'({100*grand_unaligned_size/max(1,grand_tensors):.1f}%)')
        print(f'   requests: naive={grand_tensors} -> coalesced={grand_requests}')
        print(f'   dtype mix: {dict(dtypes.most_common(10))}')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    blk = 4096
    if '--block' in sys.argv:
        blk = int(sys.argv[sys.argv.index('--block') + 1])
        args = [a for a in args if a != str(blk)]
    main(args, blk)
