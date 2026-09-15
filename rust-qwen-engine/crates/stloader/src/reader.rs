//! The read engine: `io_uring` + `O_DIRECT`, depth-bounded and pipelined.
//!
//! Design notes
//! ------------
//! * One task list of `(file, offset, len)` chunks, expanded from the planned
//!   ranges. Chunks are block-aligned on both ends.
//! * A fixed pool of `depth` aligned buffers. A slot is handed back to the free
//!   list only after its completion is reaped, which is what makes buffer reuse
//!   sound without a copy.
//! * Short reads are expected at end-of-file (the final chunk is rounded up
//!   past EOF) and are counted, not treated as errors.

use std::fs::File;
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Instant;

use io_uring::{opcode, types, IoUring};

use crate::aligned::AlignedBuf;
use crate::plan::{ceil_to, floor_to, Range};

#[derive(Debug, Clone)]
pub struct ReadConfig {
    /// Number of in-flight reads.
    pub depth: usize,
    /// Bytes per read request. Rounded up to a multiple of `block`.
    pub chunk_bytes: usize,
    /// Open with `O_DIRECT`, bypassing the page cache.
    pub direct: bool,
    /// Logical block size for alignment (4096 on this hardware).
    pub block: usize,
}

impl Default for ReadConfig {
    fn default() -> Self {
        Self { depth: 32, chunk_bytes: 1 << 20, direct: true, block: 4096 }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ReadStats {
    pub bytes: u64,
    pub requests: u64,
    pub short_reads: u64,
    pub seconds: f64,
}

impl ReadStats {
    pub fn gib(&self) -> f64 {
        self.bytes as f64 / 1024f64.powi(3)
    }

    pub fn gbps(&self) -> f64 {
        if self.seconds > 0.0 {
            self.bytes as f64 / 1e9 / self.seconds
        } else {
            0.0
        }
    }
}

fn open_file(p: &Path, direct: bool) -> io::Result<File> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    if direct {
        o.custom_flags(libc::O_DIRECT);
    }
    o.open(p)
}

/// Read every planned range, discarding the bytes.
///
/// This measures the achievable load throughput; the byte content is not
/// retained. Use [`read_small_range`] to verify correctness of a specific
/// tensor's bytes.
pub fn read_files(
    files: &[PathBuf],
    plan: &[(usize, Range)],
    cfg: &ReadConfig,
) -> io::Result<ReadStats> {
    // A plan built for a different file set must be rejected, not indexed into.
    for (fi, r) in plan {
        if *fi >= files.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("plan references shard {fi} but only {} files given", files.len()),
            ));
        }
        if r.len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("plan contains a zero-length range at offset {}", r.offset),
            ));
        }
    }
    if plan.is_empty() {
        return Ok(ReadStats::default());
    }

    let handles: Vec<File> = files
        .iter()
        .map(|p| open_file(p, cfg.direct))
        .collect::<io::Result<_>>()?;
    let fds: Vec<i32> = handles.iter().map(|f| f.as_raw_fd()).collect();

    let block = cfg.block.max(1) as u64;
    let chunk = {
        let c = cfg.chunk_bytes.max(cfg.block).max(1) as u64;
        ceil_to(c, block)
    };

    // Expand planned ranges into fixed-size, block-aligned chunks.
    let mut jobs: Vec<(usize, u64, usize)> = Vec::new();
    for (fi, r) in plan {
        let end = r.offset + r.len;
        let mut off = r.offset;
        while off < end {
            let want = chunk.min(end - off);
            // Round the final chunk up so the request stays block-aligned;
            // reading past EOF yields a short read, which is handled below.
            let len = ceil_to(want, block) as usize;
            jobs.push((*fi, off, len));
            off += want;
        }
    }

    let depth = cfg.depth.max(1);
    let mut bufs: Vec<AlignedBuf> = (0..depth)
        .map(|_| AlignedBuf::new(chunk as usize, cfg.block))
        .collect();
    let mut free: Vec<usize> = (0..depth).rev().collect();
    let mut expect = vec![0usize; depth];

    let entries = depth.next_power_of_two().max(2);
    let mut ring = IoUring::new(entries as u32)?;

    let mut stats = ReadStats::default();
    let mut next = 0usize;
    let mut inflight = 0usize;

    let t0 = Instant::now();

    loop {
        // Fill the submission queue from the free buffer pool.
        {
            let mut sq = ring.submission();
            while next < jobs.len() && !free.is_empty() && !sq.is_full() {
                let slot = free.pop().expect("free is non-empty");
                let (fi, off, len) = jobs[next];
                let entry =
                    opcode::Read::new(types::Fd(fds[fi]), bufs[slot].as_mut_ptr(), len as u32)
                        .offset(off)
                        .build()
                        .user_data(slot as u64);
                // SAFETY: `entry` points at bufs[slot]; that slot stays out of
                // `free` until this request's completion is reaped, so the
                // buffer cannot be reused or dropped while the kernel owns it.
                match unsafe { sq.push(&entry) } {
                    Ok(()) => {
                        expect[slot] = len;
                        next += 1;
                        inflight += 1;
                    }
                    Err(_) => {
                        free.push(slot);
                        break;
                    }
                }
            }
        }

        if inflight == 0 {
            break;
        }

        // EINTR is retryable and must not be reported as a read failure.
        loop {
            match ring.submit_and_wait(1) {
                Ok(_) => break,
                Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                Err(e) => return Err(e),
            }
        }

        {
            let mut cq = ring.completion();
            for cqe in cq.by_ref() {
                let slot = cqe.user_data() as usize;
                let res = cqe.result();
                if res < 0 {
                    return Err(io::Error::from_raw_os_error(-res));
                }
                let n = res as usize;
                if n < expect[slot] {
                    stats.short_reads += 1;
                }
                stats.bytes += n as u64;
                stats.requests += 1;
                free.push(slot);
                inflight -= 1;
            }
        }
    }

    stats.seconds = t0.elapsed().as_secs_f64();
    Ok(stats)
}

/// Read a small byte range and return it, honouring `O_DIRECT` alignment.
///
/// Intended for verification: read the same tensor range with and without
/// `O_DIRECT` and compare the bytes.
pub fn read_small_range(
    path: &Path,
    offset: u64,
    len: usize,
    direct: bool,
    block: usize,
) -> io::Result<Vec<u8>> {
    let f = open_file(path, direct)?;
    let block = block.max(1) as u64;
    let a0 = floor_to(offset, block);
    let pad = (offset - a0) as usize;
    let total = ceil_to((pad + len) as u64, block) as usize;

    let mut buf = AlignedBuf::new(total, block as usize);
    let fd = f.as_raw_fd();
    let mut done = 0usize;
    while done < total {
        // SAFETY: the destination has `total - done` bytes remaining at `done`.
        let n = unsafe {
            libc::pread(
                fd,
                buf.as_mut_ptr().add(done) as *mut libc::c_void,
                total - done,
                (a0 + done as u64) as libc::off_t,
            )
        };
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        if n == 0 {
            break; // EOF
        }
        done += n as usize;
    }
    if done < pad + len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("{}: wanted {} bytes at {}, got {}", path.display(), len, offset, done.saturating_sub(pad)),
        ));
    }
    Ok(buf.as_slice()[pad..pad + len].to_vec())
}
