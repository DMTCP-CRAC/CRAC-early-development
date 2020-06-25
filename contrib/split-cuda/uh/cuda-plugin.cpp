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
#endif
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <vector>
#include <algorithm>
#include <map>

#include "common.h"
#include "dmtcp.h"
// #include "dmtcp_dlsym.h"
#include "config.h"
#include "jassert.h"
#include "procmapsarea.h"
#include "getmmap.h"
#include "util.h"
#include "log_and_replay.h"
#include "mmap-wrapper.h"
// #include "device_heap_util.h"
#include "upper-half-wrappers.h"

// #define _real_dlsym NEXT_FNC(dlsym)
// #define _real_dlopen NEXT_FNC(dlopen)
// #define _real_dlerror NEXT_FNC(dlerror)

#define DEV_NVIDIA_STR "/dev/nvidia"

using namespace dmtcp;
std::map<void *, lhckpt_pages_t>  lh_pages_maps;
void * lh_ckpt_mem_addr = NULL;
size_t lh_ckpt_mem_size = 0;
int pagesize = sysconf(_SC_PAGESIZE);
GetMmappedListFptr_t fnc = NULL;
dmtcp::vector<MmapInfo_t> merged_uhmaps;
UpperHalfInfo_t uhInfo;

static bool skipWritingTextSegments = false;
extern "C" pid_t dmtcp_get_real_pid();
/* This function returns a range of zero or non-zero pages. If the first page
 * is non-zero, it searches for all contiguous non-zero pages and returns them.
 * If the first page is all-zero, it searches for contiguous zero pages and
 * returns them.
 */
static void
mtcp_get_next_page_range(Area *area, size_t *size, int *is_zero)
{
  char *pg;
  char *prevAddr;
  size_t count = 0;
  const size_t one_MB = (1024 * 1024);

  if (area->size < one_MB) {
    *size = area->size;
    *is_zero = 0;
    return;
  }
  *size = one_MB;
  *is_zero = Util::areZeroPages(area->addr, one_MB / MTCP_PAGE_SIZE);
  prevAddr = area->addr;
  for (pg = area->addr + one_MB;
       pg < area->addr + area->size;
       pg += one_MB) {
    size_t minsize = MIN(one_MB, (size_t)(area->addr + area->size - pg));
    if (*is_zero != Util::areZeroPages(pg, minsize / MTCP_PAGE_SIZE)) {
      break;
    }
    *size += minsize;
    if (*is_zero && ++count % 10 == 0) { // madvise every 10MB
      if (madvise(prevAddr, area->addr + *size - prevAddr,
                  MADV_DONTNEED) == -1) {
        JNOTE("error doing madvise(..., MADV_DONTNEED)")
          (JASSERT_ERRNO) ((void *)area->addr) ((int)*size);
        prevAddr = pg;
      }
    }
  }
}

static void
mtcp_write_non_rwx_and_anonymous_pages(int fd, Area *orig_area)
{
  Area area = *orig_area;

  /* Now give read permission to the anonymous/[heap]/[stack]/[stack:XXX] pages
   * that do not have read permission. We should remove the permission
   * as soon as we are done writing the area to the checkpoint image
   *
   * NOTE: Changing the permission here can results in two adjacent memory
   * areas to become one (merged), if they have similar permissions. This can
   * results in a modified /proc/self/maps file. We shouldn't get affected by
   * the changes because we are going to remove the PROT_READ later in the
   * code and that should reset the /proc/self/maps files to its original
   * condition.
   */

  JASSERT(orig_area->name[0] == '\0' || (strcmp(orig_area->name,
                                                "[heap]") == 0) ||
          (strcmp(orig_area->name, "[stack]") == 0) ||
          (Util::strStartsWith(area.name, "[stack:XXX]")));

  if ((orig_area->prot & PROT_READ) == 0) {
    JASSERT(mprotect(orig_area->addr, orig_area->size,
                     orig_area->prot | PROT_READ) == 0)
      (JASSERT_ERRNO) (orig_area->size) ((void *)orig_area->addr)
    .Text("error adding PROT_READ to mem region");
  }

  while (area.size > 0) {
    size_t size;
    int is_zero;
    Area a = area;
    if (dmtcp_infiniband_enabled && dmtcp_infiniband_enabled()) {
      size = area.size;
      is_zero = 0;
    } else {
      mtcp_get_next_page_range(&a, &size, &is_zero);
    }

    a.properties = is_zero ? DMTCP_ZERO_PAGE : 0;
    a.size = size;

    Util::writeAll(fd, &a, sizeof(a));
    if (!is_zero) {
      Util::writeAll(fd, a.addr, a.size);
    } else {
      if (madvise(a.addr, a.size, MADV_DONTNEED) == -1) {
        JNOTE("error doing madvise(..., MADV_DONTNEED)")
          (JASSERT_ERRNO) ((void *)a.addr) ((int)a.size);
      }
    }
    area.addr += size;
    area.size -= size;
  }

  /* Now remove the PROT_READ from the area if it didn't have it originally
  */
  if ((orig_area->prot & PROT_READ) == 0) {
    JASSERT(mprotect(orig_area->addr, orig_area->size, orig_area->prot) == 0)
      (JASSERT_ERRNO) ((void *)orig_area->addr) (orig_area->size)
    .Text("error removing PROT_READ from mem region.");
  }
}

static void
writememoryarea(int fd, Area *area, int stack_was_seen)
{
  void *addr = area->addr;

  if (!(area->flags & MAP_ANONYMOUS)) {
    JTRACE("save region") (addr) (area->size) (area->name) (area->offset);
  } else if (area->name[0] == '\0') {
    JTRACE("save anonymous") (addr) (area->size);
  } else {
    JTRACE("save anonymous") (addr) (area->size) (area->name) (area->offset);
  }

  if ((area->name[0]) == '\0') {
    char *brk = (char *)sbrk(0);
    if (brk > area->addr && brk <= area->addr + area->size) {
      strcpy(area->name, "[heap]");
    }
  }

  if (area->size == 0) {
    /* Kernel won't let us munmap this.  But we don't need to restore it. */
    JTRACE("skipping over [stack] segment (not the orig stack)")
      (addr) (area->size);
  } else if (0 == strcmp(area->name, "[vsyscall]") ||
             0 == strcmp(area->name, "[vectors]") ||
             0 == strcmp(area->name, "[vvar]") ||
             0 == strcmp(area->name, "[vdso]")) {
    JTRACE("skipping over memory special section")
      (area->name) (addr) (area->size);
  } else if (area->prot == 0 ||
             (area->name[0] == '\0' &&
              ((area->flags & MAP_ANONYMOUS) != 0) &&
              ((area->flags & MAP_PRIVATE) != 0))) {
    /* Detect zero pages and do not write them to ckpt image.
     * Currently, we detect zero pages in non-rwx mapping and anonymous
     * mappings only
     */
    mtcp_write_non_rwx_and_anonymous_pages(fd, area);
  } else {
    /* Anonymous sections need to have their data copied to the file,
     *   as there is no file that contains their data
     * We also save shared files to checkpoint file to handle shared memory
     *   implemented with backing files
     */
    JASSERT((area->flags & MAP_ANONYMOUS) || (area->flags & MAP_SHARED));

    if (skipWritingTextSegments && (area->prot & PROT_EXEC)) {
      area->properties |= DMTCP_SKIP_WRITING_TEXT_SEGMENTS;
      Util::writeAll(fd, area, sizeof(*area));
      JTRACE("Skipping over text segments") (area->name) ((void *)area->addr);
    } else {
      Util::writeAll(fd, area, sizeof(*area));
      Util::writeAll(fd, area->addr, area->size);
    }
  }
}


// Returns true if needle is in the haystack
static inline int
regionContains(const void *haystackStart,
               const void *haystackEnd,
               const void *needleStart,
               const void *needleEnd)
{
  return needleStart >= haystackStart && needleEnd <= haystackEnd;
}

bool isMergeable(MmapInfo_t first, MmapInfo_t second) {
  void * first_end_addr = (void *)((uint64_t)first.addr + first.len);
  if (first_end_addr == second.addr) {
    return true;
  }
  return false;
}

void getAndMergeUhMaps()
{
  if (lhInfo.lhMmapListFptr && fnc == NULL) {
    fnc = (GetMmappedListFptr_t) lhInfo.lhMmapListFptr;
    int numUhRegions = 0;
    std::vector<MmapInfo_t> uh_mmaps = fnc(&numUhRegions);

    // merge the entries if two entries are continous
    merged_uhmaps.push_back(uh_mmaps[0]);
    for(size_t i = 1; i < uh_mmaps.size(); i++) {
      MmapInfo_t last_merged = merged_uhmaps.back();
      if (isMergeable(last_merged, uh_mmaps[i])) {
        MmapInfo_t merged_item;
        merged_item.addr = last_merged.addr;
        merged_item.len = last_merged.len + uh_mmaps[i].len;
        merged_uhmaps.pop_back();
        merged_uhmaps.push_back(merged_item);
      } else {
        // insert uh_maps[i] to the merged list as a new item
        merged_uhmaps.push_back(uh_mmaps[i]);
      }
    }
    // TODO: print the content once
  }
}

/*
  This function checks whether we should skip the region or checkpoint fully or
  Partially.
  The idea is that we are recording each mmap by upper-half. So, all the
  ckpt'ble area
*/

#undef dmtcp_skip_memory_region_ckpting
EXTERNC int
dmtcp_skip_memory_region_ckpting(ProcMapsArea *area, int fd, int stack_was_seen)
{
  JNOTE("In skip area");
  ssize_t rc = 1;
  if (strstr(area->name, "vvar") ||
    strstr(area->name, "vdso") ||
    strstr(area->name, "vsyscall") ||
    strstr(area->name, DEV_NVIDIA_STR)) {
    return rc; // skip this region
  }

  // get and merge uh maps
  getAndMergeUhMaps();

  // smaller than smallest uhmaps or greater than largest address
  if ((area->endAddr < merged_uhmaps[0].addr) || \
      (area->addr > (void *)((VA)merged_uhmaps.back().addr + \
                              merged_uhmaps.back().len))) {
    return rc;
  }

  // Don't skip the lh_ckpt_region
  if (lh_ckpt_mem_addr && area->addr == lh_ckpt_mem_addr) {
    return 0;
  }

  size_t i = 0;
  while (i < merged_uhmaps.size()) {
    void *uhMmapStart = merged_uhmaps[i].addr;
    void *uhMmapEnd = (VA)merged_uhmaps[i].addr + merged_uhmaps[i].len;

    if (regionContains(uhMmapStart, uhMmapEnd, area->addr, area->endAddr)) {
      JNOTE ("Case 1 detected") ((void*)area->addr) ((void*)area->endAddr) \
        (uhMmapStart) (uhMmapEnd);
      return 0; // checkpoint this region
    } else if ((area->addr < uhMmapStart) && (uhMmapStart < area->endAddr) \
              && (area->endAddr <= uhMmapEnd)) {
      JNOTE ("Case 2 detected") ((void*)area->addr) ((void*)area->endAddr) \
        (uhMmapStart) (uhMmapEnd);

      // skip the region above the uhMmapStart
      area->addr = (VA)uhMmapStart;
      area->size = area->endAddr - area->addr;
      JNOTE ("Case 2: area to checkpoint") ((void*)area->addr) \
        ((void*)area->endAddr) (area->size);
      return 0; // checkpoint but values changed
    } else if ((uhMmapStart <= area->addr) && (area->addr < uhMmapEnd)) {
      // check the next element in the merged list if it contains in the area
      // TODO: handle that case
      //
      //  traverse until uhmap's start addr is bigger than the area -> endAddr
      //  rc = number of the item in the  array
      //
      JNOTE("Case 3: detected") ((void*)area->addr) ((void *)area->endAddr) \
        (area->size);
//      int dummy=1; while(dummy);
      ProcMapsArea newArea = *area;
      newArea.endAddr = (VA)uhMmapEnd;
      newArea.size = newArea.endAddr - newArea.addr;
      writememoryarea(fd, &newArea, stack_was_seen);
      // whiteAreas[count++] = newArea;
      while(i < merged_uhmaps.size()-1 \
            && merged_uhmaps[++i].addr < area->endAddr)
      {
        // TODO: Update the area after each writememoryarea
        // remove the merged uhmaps node when it's ckpt'ed
        ProcMapsArea newArea = *area;
        uhMmapStart = merged_uhmaps[i].addr;
        uhMmapEnd = (VA)merged_uhmaps[i].addr + merged_uhmaps[i].len;
        if(regionContains(area->addr, area->endAddr, uhMmapStart, uhMmapEnd)) {
          newArea.addr = (VA)uhMmapStart;
          newArea.endAddr = (VA)uhMmapEnd;
          newArea.size = newArea.endAddr - newArea.addr;
          writememoryarea(fd, &newArea, stack_was_seen);
        } else {
          newArea.addr = (VA)uhMmapStart;
          newArea.size = newArea.endAddr - newArea.addr;
          writememoryarea(fd, &newArea, stack_was_seen);
          return 1;
        }
      }
    } else if (regionContains(area->addr, area->endAddr,
                              uhMmapStart, uhMmapEnd)) {
      JNOTE("Case 4: detected") ((void*)area->addr) 
        ((void *)area->endAddr) (area->size);
      fflush(stdout);
      // TODO: this usecase is not completed; fix it later
      // int dummy = 1; while(dummy);
      // JNOTE("skipping the region partially ") (area->addr) (area->endAddr)
      //   (area->size) (array[i].len);
      area->addr = (VA)uhMmapStart;
      area->endAddr = (VA)uhMmapEnd;
      area->size = area->endAddr - area->addr;
      rc = 2; // skip partially
      break;
    }
    i++;
  }
  return 1;
}
/*
void init()
{
  // typedef void (*cudaRegisterAllFptr_t) ();
  void * dlsym_handle = _real_dlopen(NULL, RTLD_NOW);
  JASSERT(dlsym_handle) (_real_dlerror());
  void * cudaRegisterAllFptr = _real_dlsym(dlsym_handle, "_ZL24__sti____cudaRegisterAllv");
  JASSERT(cudaRegisterAllFptr) (_real_dlerror());
  JNOTE("found symbol") (cudaRegisterAllFptr);
}
*/

void save_lh_pages_to_memory()
{
  // get the Lower-half page maps
  lh_pages_maps = getLhPageMaps();
  /*
  // add the device heap entry to lh_pages_maps
  size_t cudaDeviceHeapSize = 0;
  cudaDeviceGetLimit(&cudaDeviceHeapSize, cudaLimitMallocHeapSize);

  JASSERT(lhInfo.lhGetDeviceHeapFptr) ("GetDeviceHeapFptr is not set up");
  GetDeviceHeapPtrFptr_t func = (GetDeviceHeapPtrFptr_t) lhInfo.lhGetDeviceHeapFptr;
  void *mallocPtr = func();

  size_t actualHeapSize = (size_t)((VA)ROUND_UP(mallocPtr) - (VA)lhInfo.lhDeviceHeap);
  JASSERT(actualHeapSize > 0) (mallocPtr) (lhInfo.lhDeviceHeap);
  lhckpt_pages_t page = {CUDA_HEAP_PAGE, lhInfo.lhDeviceHeap, actualHeapSize};
  lh_pages_maps[lhInfo.lhDeviceHeap] = page; */

  size_t total_size = sizeof(int);
  for (auto lh_page : lh_pages_maps) {
    // printf("\n Address = %p with size = %lu", lh_page.first,
    // lh_page.second.mem_len);
    // lhckpt_pages_t
    total_size += lh_page.second.mem_len + sizeof(lh_page.second);
  }
  if (total_size > 0) {
    // round up to the page size
    total_size = ((total_size + pagesize - 1) & ~(pagesize - 1));
    // mmap a region in the process address space big enough for the structure
    // + data + initial_guard_page
    void *addr = mmap(NULL, pagesize + total_size + pagesize,
                      PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    JASSERT(addr != MAP_FAILED) (addr) (JASSERT_ERRNO);
    JASSERT(mprotect(addr, pagesize, PROT_EXEC) != -1)(addr)(JASSERT_ERRNO);
    addr = (void *)((VA)addr + pagesize);
    JASSERT(mprotect((void *)((VA)addr+total_size), pagesize, PROT_EXEC) != -1)\
      (addr)(JASSERT_ERRNO)(total_size)(pagesize);

    // make this address and size available to dmtcp_skip_memory_region
    lh_ckpt_mem_addr = addr;
    lh_ckpt_mem_size = total_size;

    size_t count = 0;
    int total_entries = lh_pages_maps.size();
    memcpy(((VA)addr + count), &total_entries, sizeof total_entries);
    count += sizeof(total_entries);
    // mprotect with read permission on the page
    /* Que: should we change the read permission of the entire cuda malloc'ed
      region at once? So that when we mprotect back to the ---p permission, we
      don't see many entries in /proc/pid/maps file even if the perms are same.
    */
    for (auto lh_page : lh_pages_maps) {
      void *mem_addr = lh_page.second.mem_addr;
      size_t mem_len = lh_page.second.mem_len;
      // copy the metadata and data to the new mmap'ed region
      void * dest = memcpy(((VA)addr + count), (void *)&lh_page.second,
                  sizeof(lh_page.second));
      JASSERT(dest == ((VA)addr + count))("memcpy failed") (addr)
              (dest) (count) (sizeof(lh_page.second)) (JASSERT_ERRNO);
      count += sizeof(lh_page.second);
      // copy the actual data
      switch (lh_page.second.mem_type) {
        case (CUDA_MALLOC_PAGE):
        case (CUDA_UVM_PAGE):
        {
          cudaMemcpy(((VA)addr + count), mem_addr, mem_len, \
                     cudaMemcpyDeviceToHost);
          break;
        } /*
        case (CUDA_HEAP_PAGE):
        {
          void *__cudaPtr = NULL;
          void * deviceHeapStart = mem_addr;
          size_t heapSize = mem_len;
          cudaMalloc(&__cudaPtr, heapSize);
          JASSERT(lhInfo.lhCopyToCudaPtrFptr)
                 ("copyFromCudaPtrFptr is not set up");
          CopyToCudaPtrFptr_t func1 =
                   (CopyToCudaPtrFptr_t) lhInfo.lhCopyToCudaPtrFptr;
          func1(__cudaPtr, deviceHeapStart, heapSize);
          cudaMemcpy(((VA)addr + count),
                      __cudaPtr,
                      mem_len, cudaMemcpyDeviceToHost);
          cudaFree(__cudaPtr);
          break;
        } */
        default:
        {
          JASSERT(false) ("page type unkown");
        }
      }
      // JASSERT(dest == (void *)((uint64_t)addr + count))("memcpy failed")
      //  (addr) (count) (mem_addr) (mem_len);
      count += mem_len;
    }
  }
}

void pre_ckpt()
{
  /**/
  disableLogging();
  cudaDeviceSynchronize();
  save_lh_pages_to_memory();
  enableLogging();
}

// Writes out the lhinfo global object to a file. Returns 0 on success,
// -1 on failure.
static void
writeUhInfoToFile()
{
  char filename[100];
  snprintf(filename, 100, "./uhInfo_%d", getpid());
  int fd = open(filename, O_WRONLY | O_CREAT, 0644);
  JASSERT (fd != -1) ("Could not create uhaddr.bin file.") (JASSERT_ERRNO);

  size_t rc = write(fd, &uhInfo, sizeof(uhInfo));
  JASSERT(rc >= sizeof(uhInfo))("Wrote fewer bytes than expected to uhaddr.bin")
    (JASSERT_ERRNO);
  close(fd);
}

// sets up upper-half info for the lower-half to use on the restart
static void
setupUpperHalfInfo()
{
  GetEndOfHeapFptr_t func = (GetEndOfHeapFptr_t) lhInfo.uhEndofHeapFptr;
  uhInfo.uhEndofHeap = (void *)func();
  uhInfo.lhPagesRegion = (void *)lh_ckpt_mem_addr;
  uhInfo.cudaLogVectorFptr = (void *)&getCudaCallsLog;
  // FIXME: We'll just write out the uhInfo object to a file; the lower half
  // will read this file to figure out the information. This is ugly
  // but will work for now.
  unsigned long addr = 0;
  syscall(SYS_arch_prctl, ARCH_GET_FS, &addr);
  JNOTE("upper-half FS") ((void *)addr);
  JNOTE("uhInfo") ((void *)&uhInfo) (uhInfo.uhEndofHeap) (uhInfo.lhPagesRegion)
    (uhInfo.cudaLogVectorFptr);
  writeUhInfoToFile();
}

void resume()
{
  // unmap the region we mapped it earlier
  if (lh_ckpt_mem_addr != NULL && lh_ckpt_mem_size > 0)
  {
     JASSERT(munmap(lh_ckpt_mem_addr, lh_ckpt_mem_size) != -1)
            ("munmap failed!") (lh_ckpt_mem_addr) (lh_ckpt_mem_size);
     JASSERT(munmap((VA)lh_ckpt_mem_addr - pagesize, pagesize) != -1)
            ("munmap failed!") ((VA)lh_ckpt_mem_addr - pagesize) (pagesize);
  } else {
    JTRACE("no memory region was allocated earlier")
          (lh_ckpt_mem_addr) (lh_ckpt_mem_size);
  }
  setupUpperHalfInfo();
}

void restart()
{
  reset_wrappers();
  initialize_wrappers();
  // fix lower-half fs
  unsigned long addr = 0;
  syscall(SYS_arch_prctl, ARCH_GET_FS, &addr);
  memcpy((long *)((VA)lhInfo.lhFsAddr+40), (long *)(addr+40), sizeof(long));
  JNOTE("upper-half FS") ((void *)addr);
  JNOTE("lower-half FS") ((void *)lhInfo.lhFsAddr);
}

static void
cuda_plugin_event_hook(DmtcpEvent_t event, DmtcpEventData_t *data)
{
  switch (event) {
    case DMTCP_EVENT_INIT:
    {
      JTRACE("*** DMTCP_EVENT_INIT");
      JTRACE("Plugin intialized");
      break;
    }
    case DMTCP_EVENT_EXIT:
    {
      JTRACE("*** DMTCP_EVENT_EXIT");
      break;
    }
    case DMTCP_EVENT_PRECHECKPOINT:
    {
      pre_ckpt();
      break;
    }
    case DMTCP_EVENT_RESUME:
    {
      resume();
      break;
    }
    case DMTCP_EVENT_RESTART:
    {
      restart();
      break;
    }
    default:
      break;
  }
}

/*
static DmtcpBarrier cudaPluginBarriers[] = {
  { DMTCP_GLOBAL_BARRIER_PRE_CKPT, pre_ckpt, "checkpoint" },
  { DMTCP_GLOBAL_BARRIER_RESUME, resume, "resume" },
  { DMTCP_GLOBAL_BARRIER_RESTART, restart, "restart" }
};
*/
DmtcpPluginDescriptor_t cuda_plugin = {
  DMTCP_PLUGIN_API_VERSION,
  PACKAGE_VERSION,
  "cuda_plugin",
  "DMTCP",
  "dmtcp@ccs.neu.edu",
  "Cuda Split Plugin",
  cuda_plugin_event_hook
};
//  DMTCP_DECL_BARRIERS(cudaPluginBarriers),

DMTCP_DECL_PLUGIN(cuda_plugin);
