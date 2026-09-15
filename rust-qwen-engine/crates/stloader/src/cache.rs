//! Page-cache control and inspection.
//!
//! Cold-cache benchmarking is only meaningful when the kernel is not serving
//! the read from RAM, so we both evict (`posix_fadvise(DONTNEED)`) and verify
//! the eviction actually took effect (`mincore`).

use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

pub fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v <= 0 {
        4096
    } else {
        v as usize
    }
}

/// Advise the kernel to drop cached pages for every path, then flush.
pub fn drop_page_cache(paths: &[PathBuf]) -> io::Result<()> {
    for p in paths {
        let f = File::open(p)?;
        let fd = f.as_raw_fd();
        // SAFETY: fd is a valid open descriptor for the duration of the call.
        let rc = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
        if rc != 0 {
            return Err(io::Error::from_raw_os_error(rc));
        }
    }
    // SAFETY: no preconditions.
    unsafe { libc::sync() };
    Ok(())
}

const PROBE_WINDOW_BYTES: usize = 1 << 20;
const MAX_PROBE_PAGES: usize = 16384;
const MAX_PROBE_WINDOWS: usize = 64;

/// Fraction of `path` currently resident in the page cache.
///
/// Samples bounded windows rather than every page, so the cost is O(windows).
/// Returns `-1.0` when the probe cannot run, so "unknown" is distinguishable
/// from "cold".
pub fn resident_ratio(path: &Path) -> io::Result<f64> {
    let size = std::fs::metadata(path)?.len();
    if size == 0 {
        return Ok(-1.0);
    }
    let page = page_size();
    let size = size as usize;
    let window = PROBE_WINDOW_BYTES.min(size).max(page);
    let nwin = size.div_ceil(window).clamp(1, MAX_PROBE_WINDOWS);
    let step = window.max(size / nwin);

    let f = File::open(path)?;
    let fd = f.as_raw_fd();

    let mut resident = 0usize;
    let mut total = 0usize;
    let mut off = 0usize;

    while off < size && total < MAX_PROBE_PAGES {
        let mut len = window.min(size - off);
        len -= len % page;
        if len == 0 {
            break;
        }
        // SAFETY: mmap of a valid read-only fd with a page-aligned offset.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                off as libc::off_t,
            )
        };
        if addr == libc::MAP_FAILED {
            break;
        }

        let npages = len / page;
        let mut vec = vec![0u8; npages];
        // SAFETY: addr/len describe a live mapping; vec has one byte per page.
        let rc = unsafe {
            libc::mincore(addr, len, vec.as_mut_ptr() as *mut libc::c_uchar)
        };
        if rc == 0 {
            resident += vec.iter().filter(|b| *b & 1 == 1).count();
            total += npages;
        }
        // SAFETY: same mapping returned by mmap above.
        unsafe { libc::munmap(addr, len) };
        off += step;
    }

    if total == 0 {
        Ok(-1.0)
    } else {
        Ok(resident as f64 / total as f64)
    }
}

/// Minimum residency across all paths, or `-1.0` if unknown.
pub fn min_resident_ratio(paths: &[PathBuf]) -> f64 {
    let mut min = f64::INFINITY;
    for p in paths {
        match resident_ratio(p) {
            Ok(r) if r >= 0.0 => min = min.min(r),
            _ => return -1.0,
        }
    }
    if min.is_finite() {
        min
    } else {
        -1.0
    }
}

/// Resident set size of this process, in bytes.
pub fn peak_rss_bytes() -> u64 {
    // SAFETY: getrusage writes into the provided struct.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    if rc != 0 {
        return 0;
    }
    ru.ru_maxrss as u64 * 1024
}
