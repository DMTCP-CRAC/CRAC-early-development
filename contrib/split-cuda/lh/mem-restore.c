/****************************************************************************
 *  Copyright (C) 2019-2020 by Twinkle Jain, Rohan garg, and Gene Cooperman *
 *  jain.t@husky.neu.edu, rohgarg@ccs.neu.edu, gene@ccs.neu.edu             *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

#ifndef _GNU_SOURCE
  #define _GNU_SOURCE
#endif  // ifndef _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "common.h"
#include "kernel-loader.h"
#include "utils.h"
#include "logging.h"
#include "mtcp_sys.h"
#include "mem-restore.h"

// static int restoreFs(void *fs);

MtcpHeader mtcpHdr;

/* Steps are similar to mtcp_restart. However, unlike mtcp_restart,
   Kernel loader will not unmap it's own region. */
  // TODO: restore endOfHeap
void
restoreCheckpointImg(int ckptfd)
{
  assert(ckptfd != -1);
  //   Option step: mtcp_check_vdso(environ);
  readMtcpHeader(ckptfd);
  //   restore_brk(mtcpHdr.saved_brk, mtcpHdr.restore_addr,
  //               mtcpHdr.restore_addr + mtcpHdr.restore_size);

  restoreMemory(ckptfd);
  close(ckptfd);
}

pid_t
getUhPid(){
  return mtcpHdr.orig_pid;
}

void
returnTodmtcp()
{
  double readTime = 0.0;
  int mtcp_sys_errno = 0;
  // restore the brk
  setUhBrk(mtcpHdr.saved_brk);
  // This will restore FS and GS of the application process.
  restore_libc(&mtcpHdr.motherofall_tls_info, mtcpHdr.tls_pid_offset,
               mtcpHdr.tls_tid_offset, mtcpHdr.myinfo_gs);

  mtcpHdr.post_restart(readTime);
  DLOG(ERROR, "Restore failed! %d", mtcp_sys_errno);
}



/*****************************************************************************
 *
 *  Restore the GDT entries that are part of a thread's state
 *
 *  The kernel provides set_thread_area system call for a thread to alter a
 *  particular range of GDT entries, and it switches those entries on a
 *  per-thread basis.  So from our perspective, this is per-thread state that is
 *  saved outside user addressable memory that must be manually saved.
 *
 *****************************************************************************/
void
restore_libc(ThreadTLSInfo *tlsInfo,
             int tls_pid_offset,
             int tls_tid_offset,
             MYINFO_GS_T myinfo_gs)
{
  int mtcp_sys_errno;
  /* Every architecture needs a register to point to the current
   * TLS (thread-local storage).  This is where we set it up.
   */

  /* Patch 'struct user_desc' (gdtentrytls) of glibc to contain the
   * the new pid and tid.
   */
  DLOG(INFO, "(pid_t *)(*(unsigned long *)&(tlsInfo->gdtentrytls[0].base_addr) \
      = %p tls_pid_offset = %d",
     (pid_t *)(*(unsigned long *)&(tlsInfo->gdtentrytls[0].base_addr)), \
     tls_pid_offset);

  *(pid_t *)(*(unsigned long *)&(tlsInfo->gdtentrytls[0].base_addr)
             + tls_pid_offset) = mtcp_sys_getpid();
  if (mtcp_sys_kernel_gettid() == mtcp_sys_getpid()) {
    *(pid_t *)(*(unsigned long *)&(tlsInfo->gdtentrytls[0].base_addr)
               + tls_tid_offset) = mtcp_sys_getpid();
  }

  /* Now pass this to the kernel, so it can adjust the segment descriptor.
   * This will make different kernel calls according to the CPU architecture. */
  if (tls_set_thread_area(&(tlsInfo->gdtentrytls[0]), myinfo_gs) != 0) {
    DLOG(ERROR, "Error restoring GDT TLS entry; errno: %d\n",
         mtcp_sys_errno);
    exit(-1);
  }
}

static int
readMtcpHeader(int ckptFd)
{
  readAll(ckptFd, &mtcpHdr, sizeof mtcpHdr);
  return 1;
}

// static int
// restoreFs(void *fs)
// {
//   int rc = syscall(SYS_arch_prctl, ARCH_SET_FS, (uintptr_t)fs);
//   if (rc < 0) {
//     DLOG(ERROR, "Failed to restore fs for restart. Error: %s\n",
//          strerror(errno));
//     return -1;
//   }
//   return rc;
// }

static int
restoreMemory(int ckptfd)
{
  int rc = 0;
  Area area = {0};
  while (!rc && readAll(ckptfd, &area, sizeof area)) {
    rc = restoreMemoryRegion(ckptfd, &area);
  };
  return rc;
}

// Returns 0 on success, -1 otherwise
static int
restoreMemoryRegion(int ckptfd, Area* area)
{
  assert(area != NULL);

  ssize_t bytes = 0;
  void * mmappedat;

  // Check whether brk and end of heap are equal or not

  if (area->name[0] && strstr(area->name, "[heap]")
      && brk(NULL) != ((uint64_t)area->addr + area->size)) {
    DLOG(INFO, "WARNING: break (%d) not equal to end of heap (%p)\n",
            brk(NULL), area->addr + area->size);
  }

  // We could have replaced MAP_SHARED with MAP_PRIVATE in writeckpt.cpp
  // instead of here. But we do it this way for debugging purposes. This way,
  // readdmtcp.sh will still be able to properly list the shared memory areas.
  if (area->flags & MAP_SHARED) {
    area->flags = area->flags ^ MAP_SHARED;
    area->flags = area->flags | MAP_PRIVATE | MAP_ANONYMOUS;
  }

  /* Now mmap the data of the area into memory. */

  /* CASE MAPPED AS ZERO PAGE: */
  if ((area->properties & DMTCP_ZERO_PAGE) != 0) {
    DLOG(INFO, "restoring non-rwx anonymous area, %zx bytes at %p\n",
            area->size, area->addr);
    mmappedat = mmapWrapper(area->addr, area->size,
                            area->prot,
                            area->flags | MAP_FIXED, -1, 0);

    if (mmappedat != area->addr) {
      DLOG(ERROR, "mapping %zx bytes at %p\n", area->size, area->addr);
      exit(-1);
    }
  }

  /* CASE MAP_ANONYMOUS (usually implies MAP_PRIVATE):
   * For anonymous areas, the checkpoint file contains the memory contents
   * directly.  So mmap an anonymous area and read the file into it.
   * If file exists, turn off MAP_ANONYMOUS: standard private map
   */
  else if (area->flags & MAP_ANONYMOUS) {
    /* If there is a filename there, though, pretend like we're mapping
     * to it so a new /proc/self/maps will show a filename there like with
     * original process.  We only need read-only access because we don't
     * want to ever write the file.
     */

    int imagefd = -1;
    if (area->name[0] == '/') { /* If not null string, not [stack] or [vdso] */
      imagefd = open(area->name, O_RDONLY, 0);
      if (imagefd >= 0) {
        /* If the current file size is smaller than the original, we map the region
         * as private anonymous. Note that with this we lose the name of the region
         * but most applications may not care.
         */
        off_t curr_size = lseek(imagefd, 0, SEEK_END);
        assert(curr_size != -1);
        if (curr_size < area->offset + area->size) {
          close(imagefd);
          imagefd = -1;
          area->offset = 0;
        } else {
          area->flags ^= MAP_ANONYMOUS;
        }
      }
    }

    if (area->flags & MAP_ANONYMOUS) {
      DLOG(INFO, "restoring anonymous area, %zx  bytes at %p\n",
              area->size, area->addr);
    } else {
      DLOG(INFO, "restoring to non-anonymous area from anonymous area,"
              " %zx bytes at %p from %s + 0x%lX\n",
              area->size, area->addr, area->name, area->offset);
    }
    // NOTE: We mmap using our wrapper to track the upper half regions. This
    // enables the upper half to request for another checkpoint post restart.
    mmappedat = mmapWrapper(area->addr, area->size, area->prot,
                       MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, imagefd, area->offset);
    if (mmappedat == MAP_FAILED) {
      DLOG(ERROR, "Mapping failed for memory region (%s) at: %p of: %zu bytes. "
           "Error: %s\n", area->name, area->addr, area->size, strerror(errno));
      exit(-1);
    }
    assert(mmappedat == area -> addr);

    // Temporarily map with write permissions
    //
    int ret = mprotect(area->addr, area->size, area->prot | PROT_WRITE);

    if (ret < 0) {
      DLOG(ERROR, "Failed to add temporary  write perms for memory region (%s) at: %p "
           "of: %zu bytes. Error: %s\n",
           area->name, area->addr, area->size, strerror(errno));
      return -1;
    }
    if (imagefd >= 0 && !(area->flags & MAP_ANONYMOUS)) {
      close(imagefd);
    }
    // Read in the data
    bytes = readAll(ckptfd, area->addr, area->size);
    if (bytes < area->size) {
      DLOG(ERROR, "Read failed for memory region (%s) at: %p of: %zu bytes. "
           "Error: %s\n", area->name, area->addr, area->size, strerror(errno));
      return -1;
    }
    // Restore region permissions
   int rc = mprotect(area->addr, area->size, area->prot);
   if (rc < 0) {
     DLOG(ERROR, "Failed to restore perms for memory region (%s) at: %p "
          "of: %zu bytes. Error: %s\n",
          area->name, area->addr, area->size, strerror(errno));
     return -1;
   }
  }
  return 0;
}
