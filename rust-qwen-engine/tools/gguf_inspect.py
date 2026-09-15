#!/usr/bin/env python3
"""Minimal GGUF inspector: metadata + tensor inventory. No external deps."""
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

GGML_TYPE = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
    8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K',
    14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS',
    19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS',
    24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 30: 'BF16',
}

def main(path, show=8):
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            sys.exit('not a GGUF file')
        ver, = struct.unpack('<I', f.read(4))
        nt, = struct.unpack('<Q', f.read(8))
        nkv, = struct.unpack('<Q', f.read(8))
        print(f'== {path}')
        print(f'gguf_version={ver}  n_tensors={nt}  n_kv={nkv}')

        md = {}
        for _ in range(nkv):
            k = rstr(f)
            t, = struct.unpack('<I', f.read(4))
            md[k] = rval(f, t)

        print('\n-- metadata (interesting) --')
        interesting = [k for k in md if any(s in k for s in
                       ('architecture', 'block_count', 'head', 'embedding',
                        'rope', 'context_length', 'feed_forward', 'rms',
                        'attention', 'vocab_size', 'tokenizer.ggml.model',
                        'tokenizer.ggml.bos', 'tokenizer.ggml.eos',
                        'quantization', 'file_type', 'expert', 'ssm',
                        'conv', 'gate', 'linear'))]
        for k in sorted(interesting):
            v = md[k]
            if isinstance(v, list) and len(v) > 8:
                v = f'[{len(v)} items] {v[:6]}...'
            if isinstance(v, str) and len(v) > 100:
                v = v[:100] + '...'
            print(f'  {k} = {v}')

        print('\n-- all metadata keys --')
        print('  ' + ', '.join(sorted(md)))

        tensors = []
        for _ in range(nt):
            name = rstr(f)
            nd, = struct.unpack('<I', f.read(4))
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            dt, = struct.unpack('<I', f.read(4))
            off, = struct.unpack('<Q', f.read(8))
            tensors.append((name, dims, dt, off))

        by_type = collections.Counter(GGML_TYPE.get(t[2], f'?{t[2]}') for t in tensors)
        print('\n-- tensor types --')
        for k, v in by_type.most_common():
            print(f'  {k}: {v}')

        print(f'\n-- first {show} tensors --')
        for name, dims, dt, off in tensors[:show]:
            print(f'  {name}  dims={dims}  {GGML_TYPE.get(dt, dt)}  off={off}')

        print(f'\n-- last {show} tensors --')
        for name, dims, dt, off in tensors[-show:]:
            print(f'  {name}  dims={dims}  {GGML_TYPE.get(dt, dt)}  off={off}')

        print('\n-- name prefixes (top 30) --')
        pref = collections.Counter('.'.join(name.split('.')[:3]) for name, *_ in tensors)
        for k, v in pref.most_common(30):
            print(f'  {k}: {v}')

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
