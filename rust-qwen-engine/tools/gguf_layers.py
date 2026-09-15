#!/usr/bin/env python3
"""Group GGUF tensors by block index and classify each layer type.

Ground truth for a hybrid arch: which blk.N are linear-attention (SSM) and
which are full attention. Reads only the header/metadata/tensor-info section.
"""
import struct, sys, collections


def rstr(f):
    n, = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8', 'replace')


def rval(f, t):
    if t == 0:  return struct.unpack('<B', f.read(1))[0]
    if t == 1:  return struct.unpack('<b', f.read(1))[0]
    if t == 2:  return struct.unpack('<H', f.read(2))[0]
    if t == 3:  return struct.unpack('<h', f.read(2))[0]
    if t == 4:  return struct.unpack('<I', f.read(4))[0]
    if t == 5:  return struct.unpack('<i', f.read(4))[0]
    if t == 6:  return struct.unpack('<f', f.read(4))[0]
    if t == 7:  return struct.unpack('<?', f.read(1))[0]
    if t == 8:  return rstr(f)
    if t == 9:
        et, = struct.unpack('<I', f.read(4))
        n, = struct.unpack('<Q', f.read(8))
        return [rval(f, et) for _ in range(n)]
    if t == 10: return struct.unpack('<Q', f.read(8))[0]
    if t == 11: return struct.unpack('<q', f.read(8))[0]
    if t == 12: return struct.unpack('<d', f.read(8))[0]
    raise ValueError(f'unknown gguf value type {t}')


def main(path):
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            sys.exit('not a GGUF file')
        ver, = struct.unpack('<I', f.read(4))
        nt, = struct.unpack('<Q', f.read(8))
        nkv, = struct.unpack('<Q', f.read(8))

        md = {}
        for _ in range(nkv):
            k = rstr(f)
            t, = struct.unpack('<I', f.read(4))
            md[k] = rval(f, t)

        names = []
        for _ in range(nt):
            name = rstr(f)
            nd, = struct.unpack('<I', f.read(4))
            for _ in range(nd):
                struct.unpack('<Q', f.read(8))
            struct.unpack('<I', f.read(4))
            struct.unpack('<Q', f.read(8))
            names.append(name)

    print(f'== {path}')
    print(f'   gguf_version={ver}  tensors={nt}  arch={md.get("general.architecture")}')
    print(f'   block_count={md.get("qwen35.block_count")}  '
          f'full_attention_interval={md.get("qwen35.full_attention_interval")}')

    layers = collections.defaultdict(list)
    other = []
    for n in names:
        parts = n.split('.')
        if parts[0] == 'blk' and len(parts) > 2:
            layers[int(parts[1])].append('.'.join(parts[2:]))
        else:
            other.append(n)

    kinds = collections.Counter()
    print(f'\n-- {len(layers)} transformer blocks --')
    for i in sorted(layers):
        ts = set(layers[i])
        has_ssm = any(x.startswith('ssm_') for x in ts)
        has_full = 'attn_output' in ts
        if has_ssm:
            kind = 'linear_attn(SSM)'
        elif has_full:
            kind = 'full_attn'
        else:
            kind = 'UNKNOWN'
        kinds[kind] += 1
        print(f'  blk.{i:<3} {kind:<16} {sorted(ts)}')

    print(f'\n-- layer type counts --')
    for k, v in kinds.most_common():
        print(f'  {k}: {v}')

    print(f'\n-- non-block tensors ({len(other)}) --')
    for n in sorted(other):
        print(f'  {n}')


if __name__ == '__main__':
    main(sys.argv[1])
