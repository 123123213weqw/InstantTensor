#!/usr/bin/env python3
"""Dump exact tensor shapes for chosen blocks of a GGUF, grouped by layer role.

Usage: gguf_block_dims.py <model.gguf> [block_index ...]
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


TYPES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
         10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',
         16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',
         21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',
         28:'F64',30:'BF16'}

# GGML block byte sizes / element counts for size accounting
BLOCK = {
    'F32':(1,4),'F16':(1,2),'BF16':(1,2),
    'Q4_0':(32,18),'Q4_1':(32,20),'Q5_0':(32,22),'Q5_1':(32,24),'Q8_0':(32,34),
    'Q2_K':(256,84),'Q3_K':(256,110),'Q4_K':(256,144),'Q5_K':(256,176),
    'Q6_K':(256,210),'Q8_K':(256,292),
}


def main(path, want):
    with open(path,'rb') as f:
        assert f.read(4) == b'GGUF'
        struct.unpack('<I', f.read(4))
        nt, = struct.unpack('<Q', f.read(8))
        nkv, = struct.unpack('<Q', f.read(8))
        md = {}
        for _ in range(nkv):
            k = rstr(f); t, = struct.unpack('<I', f.read(4)); md[k] = rval(f, t)
        ts = []
        for _ in range(nt):
            name = rstr(f)
            nd, = struct.unpack('<I', f.read(4))
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            dt, = struct.unpack('<I', f.read(4))
            off, = struct.unpack('<Q', f.read(8))
            ts.append((name, dims, dt, off))

    print(f'== {path}')
    print(f'   arch={md.get("general.architecture")} '
          f'blocks={md.get("qwen35.block_count")} '
          f'full_attn_interval={md.get("qwen35.full_attention_interval")}')
    print(f'   hidden={md.get("qwen35.embedding_length")} '
          f'ffn={md.get("qwen35.feed_forward_length")} '
          f'heads={md.get("qwen35.attention.head_count")} '
          f'kv_heads={md.get("qwen35.attention.head_count_kv")} '
          f'key_len={md.get("qwen35.attention.key_length")}')
    print(f'   ssm: conv={md.get("qwen35.ssm.conv_kernel")} '
          f'group={md.get("qwen35.ssm.group_count")} '
          f'inner={md.get("qwen35.ssm.inner_size")} '
          f'state={md.get("qwen35.ssm.state_size")} '
          f'dt_rank={md.get("qwen35.ssm.time_step_rank")}')
    print(f'   rope: dim={md.get("qwen35.rope.dimension_count")} '
          f'sections={md.get("qwen35.rope.dimension_sections")} '
          f'base={md.get("qwen35.rope.freq_base")}')
    print(f'   vocab={md.get("tokenizer.ggml.tokens") and len(md["tokenizer.ggml.tokens"])}')

    byblk = collections.defaultdict(list)
    other = []
    for name, dims, dt, off in ts:
        p = name.split('.')
        if p[0] == 'blk':
            byblk[int(p[1])].append(('.'.join(p[2:]), dims, dt, off))
        else:
            other.append((name, dims, dt, off))

    for b in want:
        if b not in byblk:
            continue
        role = 'linear_attn(SSM)' if any(n.startswith('ssm_') for n,_,_,_ in byblk[b]) else 'full_attn'
        total = 0
        print(f'\n-- blk.{b}  [{role}] --')
        for name, dims, dt, off in sorted(byblk[b]):
            t = TYPES.get(dt, f'?{dt}')
            nel = 1
            for d in dims:
                nel *= d
            if t in BLOCK:
                per, nbytes = BLOCK[t]
                size = (nel // per) * nbytes
            else:
                size = 0
            total += size
            print(f'   {name:<26} dims={str(dims):<20} {t:<6} {size/1024:.1f} KiB')
        print(f'   {"TOTAL":<26} {"":<20} {"":<6} {total/1024/1024:.1f} MiB')

    print(f'\n-- non-block tensors --')
    for name, dims, dt, off in sorted(other):
        t = TYPES.get(dt, f'?{dt}')
        nel = 1
        for d in dims:
            nel *= d
        size = (nel // BLOCK[t][0]) * BLOCK[t][1] if t in BLOCK else 0
        print(f'   {name:<26} dims={str(dims):<20} {t:<6} {size/1024/1024:.1f} MiB')


if __name__ == '__main__':
    main(sys.argv[1], [int(x) for x in sys.argv[2:]] or [0])
