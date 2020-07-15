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
#define CUDA 1
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

#include "common.h"
#include "mem-restore.h"
#include "custom-loader.h"
#include "kernel-loader.h"
#include "logging.h"
#include "procmapsutils.h"
#include "trampoline_setup.h"
#include "utils.h"
#include "getmmap.h"
#include "log_and_replay.h"
// #include "device_heap_util.h"

LowerHalfInfo_t lhInfo;
UpperHalfInfo_t uhInfo;
static void readUhInfoAddr();

// Local function declarations
static void getProcStatField(enum Procstat_t , char *, size_t );
static void getStackRegion(Area *);
static void* deepCopyStack(void *, const void *, size_t,
                           const void *, const void*,
                           const DynObjInfo_t *);
static void* createNewStackForRtld(const DynObjInfo_t *);
static void* createNewHeapForRtld(const DynObjInfo_t *);
static void* getEntryPoint(DynObjInfo_t );
static void patchAuxv(ElfW(auxv_t) *, unsigned long ,
                      unsigned long , unsigned long );
static int writeLhInfoToFile();
static int setupLowerHalfInfo();
static void printUsage();
static void printRestartUsage();
// Global functions

// This function loads in ld.so, sets up a separate stack for it, and jumps
// to the entry point of ld.so
void
runRtld()
{
  int rc = -1;

  // Pointer to the ld.so entry point
  void *ldso_entrypoint = NULL;

  // Load RTLD (ld.so)
  char *ldname  = getenv("TARGET_LD");
  char *uhpreload = getenv("UH_PRELOAD");
  if (!ldname || !uhpreload) {
    printUsage();
    return;
  }

  DynObjInfo_t ldso = safeLoadLib(ldname);
  if (ldso.baseAddr == NULL || ldso.entryPoint == NULL) {
    DLOG(ERROR, "Error loading the runtime loader (%s). Exiting...\n", ldname);
    return;
  }

  DLOG(INFO, "New ld.so loaded at: %p\n", ldso.baseAddr);
  ldso_entrypoint = getEntryPoint(ldso);

  // Create new stack region to be used by RTLD
  void *newStack = createNewStackForRtld(&ldso);
  if (!newStack) {
    DLOG(ERROR, "Error creating new stack for RTLD. Exiting...\n");
    exit(-1);
  }
  DLOG(INFO, "New stack start at: %p\n", newStack);

  // Create new heap region to be used by RTLD
  void *newHeap = createNewHeapForRtld(&ldso);
  if (!newHeap) {
    DLOG(ERROR, "Error creating new heap for RTLD. Exiting...\n");
    exit(-1);
  }
  DLOG(INFO, "New heap mapped at: %p\n", newHeap);

  // insert a trampoline from ldso mmap address to mmapWrapper
  rc = insertTrampoline(ldso.mmapAddr, (void *)&mmapWrapper);
  if (rc < 0) {
    DLOG(ERROR, "Error inserting trampoline for mmap. Exiting...\n");
    exit(-1);
  }
  // insert a trampoline from ldso sbrk address to sbrkWrapper
  rc = insertTrampoline(ldso.sbrkAddr, (void *)&sbrkWrapper);
  if (rc < 0) {
    DLOG(ERROR, "Error inserting trampoline for sbrk. Exiting...\n");
    exit(-1);
  }

  // Everything is ready, let's set up the lower-half info struct for the upper
  // half to read from
  rc = setupLowerHalfInfo();
  if (rc < 0) {
    DLOG(ERROR, "Failed to set up lhinfo for the upper half. Exiting...\n");
    exit(-1);
  }

  // Change the stack pointer to point to the new stack and jump into ld.so
  // TODO: Clean up all the registers?
  asm volatile (CLEAN_FOR_64_BIT(mov %0, %%esp; )
                : : "g" (newStack) : "memory");
  asm volatile ("jmp *%0" : : "g" (ldso_entrypoint) : "memory");
}

// Local functions

static void
printUsage()
{
  DLOG(ERROR, "Usage: UH_PRELOAD=/path/to/libupperhalfwrappers.so "
          "TARGET_LD=/path/to/ld.so ./kernel-loader "
          "<target-application> [application arguments ...]\n");
}

static void
printRestartUsage()
{
  DLOG(ERROR, "Usage: ./kernel-loader --restore /path/to/ckpt.img\n");
}
//extern "C" void** __cudaRegisterFatBinary(void *fatCubin);
//extern void ** getCubinHandle();
// #define shift argv++; argc--;
int
main(int argc, char *argv[], char **environ)
{
  if (argc < 2) {
    printUsage();
    return -1;
  }
  if (strstr(argv[1], "--restore")) {
    if (argc < 3) {
      printRestartUsage();
      return -1;
    }
    int ckptFd = atoi(argv[2]);

    // setup lower-half info including cuda APIs function pointers
    int rc = setupLowerHalfInfo();
    if (rc < 0) {
      DLOG(ERROR, "Failed to set up lhinfo for the upper half. Exiting...\n");
      exit(-1);
    }
//    void * cptr=NULL;
 //   cudaMalloc(&cptr, 436*sizeof(char));
	
    //testing
   // lhInfo.new_getFatCubinHandle=(void *)&getCubinHandle;
    //
    /*
     restoreCheckpoint will
     1. read the MtcpHeader
     2. restore the memory region of the application from ckpt image.
     3. return to the plugin code of the checkpoint thread.
    */
    restoreCheckpointImg(ckptFd);
    readUhInfoAddr();
    
    logs_read_and_apply();
    
    copy_lower_half_data();
    returnTodmtcp();
    // Following line should not be reached.
    // dprintf(stderr_fd, "Restore failed!");
  }
  runRtld();
  return 0;
}
// Returns the /proc/self/stat entry in the out string (of length len)
static void
getProcStatField(enum Procstat_t type, char *out, size_t len)
{
  const char *procPath = "/proc/self/stat";
  char sbuf[1024] = {0};
  int field_counter = 0;
  char *field_str = NULL;
  int fd, num_read;

  fd = open(procPath, O_RDONLY);
  if (fd < 0) {
    DLOG(ERROR, "Failed to open %s. Error: %s\n", procPath, strerror(errno));
    return;
  }

  num_read = read(fd, sbuf, sizeof sbuf - 1);
  close(fd);
  if (num_read <= 0) return;
  sbuf[num_read] = '\0';

  field_str = strtok(sbuf, " ");
  while (field_str && field_counter != type) {
    field_str = strtok(NULL, " ");
    field_counter++;
  }

  if (field_str) {
    strncpy(out, field_str, len);
  } else {
    DLOG(ERROR, "Failed to parse %s.\n", procPath);
  }
}

// Returns the [stack] area by reading the proc maps
static void
getStackRegion(Area *stack) // OUT
{
  Area area;
  int mapsfd = open("/proc/self/maps", O_RDONLY);
  while (readMapsLine(mapsfd, &area)) {
    if (strstr(area.name, "[stack]") && area.endAddr >= (VA)&area) {
      *stack = area;
      break;
    }
  }
  close(mapsfd);
}

// Given a pointer to aux vector, parses the aux vector, and patches the
// following three entries: AT_PHDR, AT_ENTRY, and AT_PHNUM
static void
patchAuxv(ElfW(auxv_t) *av, unsigned long phnum,
          unsigned long phdr, unsigned long entry)
{
  for (; av->a_type != AT_NULL; ++av) {
    switch (av->a_type) {
      case AT_PHNUM:
        av->a_un.a_val = phnum;
        break;
      case AT_PHDR:
        av->a_un.a_val = phdr;
        break;
      case AT_ENTRY:
        av->a_un.a_val = entry;
        break;
      case AT_RANDOM:
        DLOG(NOISE, "AT_RANDOM value: 0%lx\n", av->a_un.a_val);
        break;
      default:
        break;
    }
  }
}

// Creates a deep copy of the stack region pointed to be `origStack` at the
// location pointed to be `newStack`. Returns the start-of-stack pointer
// in the new stack region.
static void*
deepCopyStack(void *newStack, const void *origStack, size_t len,
              const void *newStackEnd, const void *origStackEnd,
              const DynObjInfo_t *info)
{
  // This function assumes that this env var is set.
  assert(getenv("TARGET_LD"));
  assert(getenv("UH_PRELOAD"));

  // Return early if any pointer is NULL
  if (!newStack || !origStack ||
      !newStackEnd || !origStackEnd ||
      !info) {
    return NULL;
  }

  // First, we do a shallow copy, which is essentially, just copying the
  // bits from the original stack into the new stack.
  memcpy(newStack, origStack, len);

  // Next, turn the shallow copy into a deep copy.
  //
  // The main thing we need to do is to patch the argv and env vectors in
  // the new stack to point to addresses in the new stack region. Note that
  // the argv and env are simply arrays of pointers. The pointers point to
  // strings in other locations in the stack.

  void *origArgcAddr     = (void*)GET_ARGC_ADDR(origStackEnd);
  int  origArgc          = *(int*)origArgcAddr;
  char **origArgv        = (char**)GET_ARGV_ADDR(origStackEnd);
  const char **origEnv   = (const char**)GET_ENV_ADDR(origArgv, origArgc);

  void *newArgcAddr     = (void*)GET_ARGC_ADDR(newStackEnd);
  int  newArgc          = *(int*)newArgcAddr;
  char **newArgv        = (char**)GET_ARGV_ADDR(newStackEnd);
  const char **newEnv   = (const char**)GET_ENV_ADDR(newArgv, newArgc);
  ElfW(auxv_t) *newAuxv = GET_AUXV_ADDR(newEnv);

  // Patch the argv vector in the new stack
  //   First, set up the argv vector based on the original stack
  for (int i = 0; origArgv[i] != NULL; i++) {
    off_t argvDelta = (uintptr_t)origArgv[i] - (uintptr_t)origArgv;
    newArgv[i] = (char*)((uintptr_t)newArgv + (uintptr_t)argvDelta);
  }

  //   Next, we patch argv[0], the first argument, on the new stack
  //   to point to "/path/to/ld.so".
  //
  //   From the point of view of ld.so, it would appear as if it was called
  //   like this: $ /lib/ld.so /path/to/target.exe app-args ...
  //
  //   NOTE: The kernel loader needs to be called with at least two arguments
  //   to get a stack that is 16-byte aligned at the start. Since we want to
  //   be able to jump into ld.so with at least two arguments (ld.so and the
  //   target exe) on the new stack, we also need two arguments on the
  //   original stack.
  //
  //   If the original stack had just one argument, we would have inherited
  //   that alignment in the new stack. Trying to push in another argument
  //   (target exe) on the new stack would destroy the 16-byte alignment
  //   on the new stack. This would lead to a crash later on in ld.so.
  //
  //   The problem is that there are instructions (like, "movaps") in ld.so's
  //   code that operate on the stack memory region and require their
  //   operands to be 16-byte aligned. A non-16-byte-aligned operand (for
  //   example, the stack base pointer) leads to a general protection
  //   exception (#GP), which translates into a segfault for the user
  //   process.
  //
  //   The Linux kernel ensures that the start of stack is always 16-byte
  //   aligned. It seems like this is part of the Linux kernel x86-64 ABI.
  //   For example, see here:
  //
  //     https://elixir.bootlin.com/linux/v4.18.11/source/fs/binfmt_elf.c#L150
  //
  //     https://elixir.bootlin.com/linux/v4.18.11/source/fs/binfmt_elf.c#L288
  //
  //   (The kernel uses the STACK_ROUND macro to first set up the stack base
  //    at a 16-byte aligned address, and then pushes items on the stack.)
  //
  //   We could do something similar on the new stack region. But perhaps it's
  //   easier to just depend on the original stack having at least two args:
  //   "/path/to/kernel-loader" and "/path/to/target.exe".
  //
  //   NOTE: We don't need to patch newArgc, since the original stack,
  //   from where we would have inherited the data in the new stack, already
  //   had the correct value for origArgc. We just make argv[0] in the
  //   new stack to point to "/path/to/ld.so", instead of
  //   "/path/to/kernel-loader".
  off_t argvDelta = (uintptr_t)getenv("TARGET_LD") - (uintptr_t)origArgv;
  newArgv[0] = (char*)((uintptr_t)newArgv + (uintptr_t)argvDelta);

  // Patch the env vector in the new stack
  for (int i = 0; origEnv[i] != NULL; i++) {
    off_t envDelta = (uintptr_t)origEnv[i] - (uintptr_t)origEnv;
    newEnv[i] = (char*)((uintptr_t)newEnv + (uintptr_t)envDelta);
  }

  // Change "UH_PRELOAD" to "LD_PRELOAD". This way, upper half's ld.so
  // will preload the upper half wrapper library.
  char **newEnvPtr = (char**)newEnv;
  for (; *newEnvPtr; newEnvPtr++) {
    if (strstr(*newEnvPtr, "UH_PRELOAD")) {
      (*newEnvPtr)[0] = 'L';
      (*newEnvPtr)[1] = 'D';
      break;
    }
  }

  // The aux vector, which we would have inherited from the original stack,
  // has entries that correspond to the kernel loader binary. In particular,
  // it has these entries AT_PHNUM, AT_PHDR, and AT_ENTRY that correspond
  // to kernel-loader. So, we atch the aux vector in the new stack to
  // correspond to the new binary: the freshly loaded ld.so.
  patchAuxv(newAuxv, info->phnum,
            (uintptr_t)info->phdr,
            (uintptr_t)info->entryPoint);

printf("newArgv[-2]: %lu \n", (unsigned long)&newArgv[0]);

  // We clear out the rest of the new stack region just in case ...
  memset(newStack, 0, (size_t)((uintptr_t)&newArgv[-2] - (uintptr_t)newStack));

  // Return the start of new stack.
  return (void*)newArgcAddr;
}

// This function does three things:
//  1. Creates a new stack region to be used for initialization of RTLD (ld.so)
//  2. Deep copies the original stack (from the kernel) in the new stack region
//  3. Returns a pointer to the beginning of stack in the new stack region
static void*
createNewStackForRtld(const DynObjInfo_t *info)
{
  Area stack;
  char stackEndStr[20] = {0};
  getStackRegion(&stack);

  // 1. Allocate new stack region
  // We go through the mmap wrapper function to ensure that this gets added
  // to the list of upper half regions to be checkpointed.
  void *newStack = mmapWrapper(NULL, stack.size, PROT_READ | PROT_WRITE,
                               MAP_GROWSDOWN | MAP_PRIVATE | MAP_ANONYMOUS,
                               -1, 0);
  if (newStack == MAP_FAILED) {
    DLOG(ERROR, "Failed to mmap new stack region: %s\n", strerror(errno));
    return NULL;
  }
  DLOG(INFO, "New stack mapped at: %p\n", newStack);

  // 3. Get pointer to the beginning of the stack in the new stack region
  // The idea here is to look at the beginning of stack in the original
  // stack region, and use that to index into the new memory region. The
  // same offsets are valid in both the stack regions.
  getProcStatField(STARTSTACK, stackEndStr, sizeof stackEndStr);

  // NOTE: The kernel sets up the stack in the following format.
  //      -1(%rsp)                       Stack end for application
  //      0(%rsp)                        argc (Stack start for application)
  //      LP_SIZE(%rsp)                  argv[0]
  //      (2*LP_SIZE)(%rsp)              argv[1]
  //      ...
  //      (LP_SIZE*(argc))(%rsp)         NULL
  //      (LP_SIZE*(argc+1))(%rsp)       envp[0]
  //      (LP_SIZE*(argc+2))(%rsp)       envp[1]
  //      ...
  //                                     NULL
  //
  // NOTE: proc-stat returns the address of argc on the stack.
  // argv[0] is 1 LP_SIZE ahead of argc, i.e., startStack + sizeof(void*)
  // Stack End is 1 LP_SIZE behind argc, i.e., startStack - sizeof(void*)
  // sizeof(unsigned long) == sizeof(void*) == 8 on x86-64
  unsigned long origStackEnd = atol(stackEndStr) - sizeof(unsigned long);
  unsigned long origStackOffset = origStackEnd - (unsigned long)stack.addr;
  unsigned long newStackOffset = origStackOffset;
  void *newStackEnd = (void*)((unsigned long)newStack + newStackOffset);

printf("origStack: %lu origStackOffset: %lu OrigStackEnd: %lu \n", (unsigned long)stack.addr, (unsigned long)origStackOffset, (unsigned long)origStackEnd);
printf("newStack: %lu newStackOffset: %lu newStackEnd: %lu \n", (unsigned long)newStack, (unsigned long)newStackOffset, (unsigned long)newStackEnd);

  // 2. Deep copy stack
  newStackEnd = deepCopyStack(newStack, stack.addr, stack.size,
                              (void*)newStackEnd, (void*)origStackEnd,
                              info);

  return newStackEnd;
}

// This function allocates a new heap for (the possibly second) ld.so.
// The initial heap size is 1 page
//
// Returns the start address of the new heap on success, or NULL on
// failure.
static void*
createNewHeapForRtld(const DynObjInfo_t *info)
{
  const uint64_t heapSize = 100 * PAGE_SIZE;

  // We go through the mmap wrapper function to ensure that this gets added
  // to the list of upper half regions to be checkpointed.
  void *addr = mmapWrapper(0, heapSize, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED) {
    DLOG(ERROR, "Failed to mmap region. Error: %s\n",
         strerror(errno));
    return NULL;
  }
  // Add a guard page before the start of heap; this protects
  // the heap from getting merged with a "previous" region.
  mprotect(addr, PAGE_SIZE, PROT_NONE);
  setUhBrk((void*)((VA)addr + PAGE_SIZE));
  setEndOfHeap((void*)((VA)addr + heapSize));
  return addr;
}

// This function returns the entry point of the ld.so executable given
// the library handle
static void*
getEntryPoint(DynObjInfo_t info)
{
  return info.entryPoint;
}

// Writes out the lhinfo global object to a file. Returns 0 on success,
// -1 on failure.
static int
writeLhInfoToFile()
{
  size_t rc = 0;
  char filename[100];
  snprintf(filename, 100, "./lhInfo_%d", getpid());
  int fd = open(filename, O_WRONLY | O_CREAT, 0644);
  if (fd < 0) {
    DLOG(ERROR, "Could not create addr.bin file. Error: %s", strerror(errno));
    return -1;
  }

  rc = write(fd, &lhInfo, sizeof(lhInfo));
  if (rc < sizeof(lhInfo)) {
    DLOG(ERROR, "Wrote fewer bytes than expected to addr.bin. Error: %s",
         strerror(errno));
    rc = -1;
  }
  close(fd);
  return rc;
}

// Sets up lower-half info struct for the upper half to read from. Returns 0
// on success, -1 otherwise
static int
setupLowerHalfInfo()
{
  lhInfo.lhSbrk = (void *)&sbrkWrapper;
  lhInfo.lhMmap = (void *)&mmapWrapper;
  lhInfo.lhMunmap = (void *)&munmapWrapper;
  lhInfo.lhDlsym = (void *)&lhDlsym;
  lhInfo.lhMmapListFptr = (void *)&getMmappedList;
  lhInfo.uhEndofHeapFptr = (void *)&getEndOfHeap;
  lhInfo.getFatCubinHandle=(void *)&fatHandle;
  // lhInfo.lhDeviceHeap = (void *)ROUND_DOWN(getDeviceHeapPtr());
  // lhInfo.lhGetDeviceHeapFptr = (void *)&getDeviceHeapPtr;
  // lhInfo.lhCopyToCudaPtrFptr = (void *)&copyToCudaPtr;
  if (syscall(SYS_arch_prctl, ARCH_GET_FS, &lhInfo.lhFsAddr) < 0) {
    DLOG(ERROR, "Could not retrieve lower half's fs. Error: %s. Exiting...\n",
         strerror(errno));
    return -1;
  }
  // FIXME: We'll just write out the lhInfo object to a file; the upper half
  // will read this file to figure out the wrapper addresses. This is ugly
  // but will work for now.
  int rc = writeLhInfoToFile();
  if (rc < 0) {
    DLOG(ERROR, "Error writing address of lhinfo to file. Exiting...\n");
    return -1;
  }
  return 0;
}

static void
readUhInfoAddr()
{
  char filename[100];
  // snprintf(filename, 100, "./uhInfo_%d", getpid());
  pid_t orig_pid = getUhPid();
  snprintf(filename, 100, "./uhInfo_%d", orig_pid);
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    printf("Could not open upper-half file for reading. %s \n", filename);
    exit(-1);
  }
  ssize_t rc = read(fd, &uhInfo, sizeof(uhInfo));
  if (rc != (ssize_t)sizeof(uhInfo)) {
    perror("Read fewer bytes than expected from uhaddr.bin.");
    exit(-1);
  }
//  unlink(UH_FILE_NAME);
//  close(fd);
}

// enum for types
// enum pages_t {
//   CUDA_MALLOC_PAGE = 0,
//   CUDA_UVM_PAGE
// };

// typedef struct Lhckpt_pages_t {
//   pages_t mem_type;
//   void * mem_addr;
//   size_t mem_len;
// }lhckpt_pages_t;

void
copy_lower_half_data() {
  void * lhpages_addr = uhInfo.lhPagesRegion;

  // read total entries count
  int total_entries = 0;
  int count = 0;
  memcpy(&total_entries, ((VA)lhpages_addr+count), sizeof (total_entries));
  count += sizeof (total_entries);
  for (int i = 0; i < total_entries; i++) {
    // read metadata of one entry
    lhckpt_pages_t lhpage_info;
    memcpy(&lhpage_info, ((VA)lhpages_addr+count), sizeof (lhpage_info));
    count += sizeof(lhpage_info);

    void *dest_addr = lhpage_info.mem_addr;
    size_t size = lhpage_info.mem_len;

    switch (lhpage_info.mem_type) {
      case (CUDA_UVM_PAGE):
      case (CUDA_MALLOC_PAGE):
      {
        // copy back the actual data
        cudaMemcpy(dest_addr, ((VA)lhpages_addr+count), size, cudaMemcpyHostToDevice);
        count += size;
        break;
      }
/*      case CUDA_HEAP_PAGE:
      {
        void *newDeviceHeapStart = (void *)ROUND_DOWN(getDeviceHeapPtr());
        void *__cudaPtr = NULL;
        void *oldDeviceHeapStart = dest_addr;
        if (oldDeviceHeapStart != newDeviceHeapStart) {
          DLOG(ERROR, "New Device heap = %p is not same as Old device heap =%p\n",
          newDeviceHeapStart, oldDeviceHeapStart);
        }
        cudaMalloc(&__cudaPtr, size);
        cudaMemcpy(__cudaPtr, ((VA)lhpages_addr+count), size, cudaMemcpyHostToDevice);
        copyFromCudaPtr(__cudaPtr, newDeviceHeapStart, size);
        char buf[8192];
        copyToCudaPtr(__cudaPtr, newDeviceHeapStart, size);
        cudaMemcpy(buf, __cudaPtr, size, cudaMemcpyDeviceToHost);
        cudaFree(__cudaPtr);
        cudaDeviceSynchronize();
        count += size;
        break;
      } */
      default:
        printf("page type not implemented\n");
        break;
    }
  }
}
