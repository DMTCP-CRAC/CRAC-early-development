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

#include <assert.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "common.h"
#include "logging.h"
#include "custom-loader.h"
#include "kernel-loader.h"
#include "trampoline_setup.h"

// #define UBUNTU 1

// Uses ELF Format.  For background, read both of:
//  * http://www.skyfree.org/linux/references/ELF_Format.pdf
//  * /usr/include/elf.h
// The kernel code is here (not recommended for a first reading):
//    https://github.com/torvalds/linux/blob/master/fs/binfmt_elf.c

static void get_elf_interpreter(int , Elf64_Addr *, char* , void* );
static void* load_elf_interpreter(int , char* , Elf64_Addr *,
                                  void * , DynObjInfo_t* );
static void* map_elf_interpreter_load_segment(int , Elf64_Phdr , void* );

// Global functions
DynObjInfo_t
safeLoadLib(const char *name)
{
  void *ld_so_addr = NULL;
  DynObjInfo_t info = {0};

  int ld_so_fd;
  Elf64_Addr cmd_entry, ld_so_entry;
  char elf_interpreter[MAX_ELF_INTERP_SZ];

  // FIXME: Do we need to make it dynamic? Is setting this required?
  // ld_so_addr = (void*)0x7ffff81d5000;
  // ld_so_addr = (void*)0x7ffff7dd7000;
  int cmd_fd = open(name, O_RDONLY);
  get_elf_interpreter(cmd_fd, &cmd_entry, elf_interpreter, ld_so_addr);
  // FIXME: The ELF Format manual says that we could pass the cmd_fd to ld.so,
  //   and it would use that to load it.
  close(cmd_fd);
  strncpy(elf_interpreter, name, sizeof elf_interpreter);

  ld_so_fd = open(elf_interpreter, O_RDONLY);
  assert(ld_so_fd != -1);
  info.baseAddr = load_elf_interpreter(ld_so_fd, elf_interpreter,
                                        &ld_so_entry, ld_so_addr, &info);
  off_t mmap_offset;
  off_t sbrk_offset;
#if UBUNTU
  char buf[256] = "/usr/lib/debug";
  buf[sizeof(buf)-1] = '\0';
  ssize_t rc = 0;
  rc = readlink(elf_interpreter, buf+strlen(buf), sizeof(buf)-strlen(buf)-1);
  if (rc != -1 && access(buf, F_OK) == 0) {
    // Debian family (Ubuntu, etc.) use this scheme to store debug symbols.
    //   http://sourceware.org/gdb/onlinedocs/gdb/Separate-Debug-Files.html
    fprintf(stderr, "Debug symbols for interpreter in: %s\n", buf);
  }
  printf("%s\n", buf);
  int debug_ld_so_fd = open(buf, O_RDONLY);
  assert(debug_ld_so_fd != -1);
  mmap_offset = get_symbol_offset(debug_ld_so_fd, buf, "mmap");
  sbrk_offset = get_symbol_offset(debug_ld_so_fd, buf, "sbrk");
  close(debug_ld_so_fd);
#else
  mmap_offset = get_symbol_offset(ld_so_fd, name, "mmap");
  sbrk_offset = get_symbol_offset(ld_so_fd, name, "sbrk");
#endif
  assert(mmap_offset);
  assert(sbrk_offset);
  info.mmapAddr = (VA)info.baseAddr + mmap_offset;
  info.sbrkAddr = (VA)info.baseAddr + sbrk_offset;
  // FIXME: The ELF Format manual says that we could pass the ld_so_fd to ld.so,
  //   and it would use that to load it.
  close(ld_so_fd);
  info.entryPoint = (void*)((unsigned long)info.baseAddr +
                            (unsigned long)cmd_entry);
  return info;
}

// Local functions
static void
get_elf_interpreter(int fd, Elf64_Addr *cmd_entry,
                    char* elf_interpreter, void *ld_so_addr)
{
  int rc;
  char e_ident[EI_NIDENT];

  rc = read(fd, e_ident, sizeof(e_ident));
  assert(rc == sizeof(e_ident));
  assert(strncmp(e_ident, ELFMAG, strlen(ELFMAG)) == 0);
  assert(e_ident[EI_CLASS] == ELFCLASS64); // FIXME:  Add support for 32-bit ELF

  // Reset fd to beginning and parse file header
  lseek(fd, 0, SEEK_SET);
  Elf64_Ehdr elf_hdr;
  rc = read(fd, &elf_hdr, sizeof(elf_hdr));
  assert(rc == sizeof(elf_hdr));
  *cmd_entry = elf_hdr.e_entry;

  // Find ELF interpreter
  int i;
  Elf64_Phdr phdr;
  int phoff = elf_hdr.e_phoff;

  lseek(fd, phoff, SEEK_SET);
  for (i = 0; i < elf_hdr.e_phnum; i++) {
    assert(i < elf_hdr.e_phnum);
    rc = read(fd, &phdr, sizeof(phdr)); // Read consecutive program headers
    assert(rc == sizeof(phdr));
#if 0 // not required for this project
    if (phdr.p_type == PT_INTERP) break;
  }
  lseek(fd, phdr.p_offset, SEEK_SET); // Point to beginning of elf interpreter
  assert(phdr.p_filesz < MAX_ELF_INTERP_SZ);
  rc = read(fd, elf_interpreter, phdr.p_filesz);
  assert(rc == phdr.p_filesz);

  DLOG(INFO, "Interpreter: %s\n", elf_interpreter);
  { char buf[256] = "/usr/lib/debug";
    buf[sizeof(buf)-1] = '\0';
    int rc = 0;
    rc = readlink(elf_interpreter, buf+strlen(buf), sizeof(buf)-strlen(buf)-1);
    if (rc != -1 && access(buf, F_OK) == 0) {
      // Debian family (Ubuntu, etc.) use this scheme to store debug symbols.
      //   http://sourceware.org/gdb/onlinedocs/gdb/Separate-Debug-Files.html
      DLOG(INFO, "Debug symbols for interpreter in: %s\n", buf);
    }
  }
#else // ifdef UBUNTU
  }
#endif // ifdef UBUNTU
}

static void*
load_elf_interpreter(int fd, char *elf_interpreter,
                     Elf64_Addr *ld_so_entry, void *ld_so_addr,
                     DynObjInfo_t *info)
{
  char e_ident[EI_NIDENT];
  int rc;
  int firstTime = 1;
  void *baseAddr = NULL;

  rc = read(fd, e_ident, sizeof(e_ident));
  assert(rc == sizeof(e_ident));
  assert(strncmp(e_ident, ELFMAG, sizeof(ELFMAG)-1) == 0);
  // FIXME:  Add support for 32-bit ELF later
  assert(e_ident[EI_CLASS] == ELFCLASS64);

  // Reset fd to beginning and parse file header
  lseek(fd, 0, SEEK_SET);
  Elf64_Ehdr elf_hdr;
  rc = read(fd, &elf_hdr, sizeof(elf_hdr));
  assert(rc == sizeof(elf_hdr));

  // Find ELF interpreter
  int phoff = elf_hdr.e_phoff;
  Elf64_Phdr phdr;
  int i;
  lseek(fd, phoff, SEEK_SET);
  for (i = 0; i < elf_hdr.e_phnum; i++ ) {
    rc = read(fd, &phdr, sizeof(phdr)); // Read consecutive program headers
    assert(rc == sizeof(phdr));
    if (phdr.p_type == PT_LOAD) {
      // PT_LOAD is the only type of loadable segment for ld.so
      if (firstTime) {
        baseAddr = map_elf_interpreter_load_segment(fd, phdr, ld_so_addr);
        firstTime = 0;
      } else {
        map_elf_interpreter_load_segment(fd, phdr, ld_so_addr);
      }
    }
  }
  info->phnum = elf_hdr.e_phnum;
  info->phdr = (VA)baseAddr + elf_hdr.e_phoff;
  return baseAddr;
}

static void*
map_elf_interpreter_load_segment(int fd, Elf64_Phdr phdr, void *ld_so_addr)
{
  static char *base_address = NULL; // is NULL on call to first LOAD segment
  static int first_time = 1;
  int prot = PROT_NONE;
  if (phdr.p_flags & PF_R)
    prot |= PROT_READ;
  if (phdr.p_flags & PF_W)
    prot |= PROT_WRITE;
  if (phdr.p_flags & PF_X)
    prot |= PROT_EXEC;
  assert(phdr.p_memsz >= phdr.p_filesz);
  // NOTE:  man mmap says:
  // For a file that is not a  multiple  of  the  page  size,  the
  // remaining memory is zeroed when mapped, and writes to that region
  // are not written out to the file.
  void *rc2;
  // Check ELF Format constraint:
  if (phdr.p_align > 1) {
    assert(phdr.p_vaddr % phdr.p_align == phdr.p_offset % phdr.p_align);
  }
  int vaddr = phdr.p_vaddr;

  int flags = MAP_PRIVATE;
  unsigned long addr = ROUND_DOWN(base_address + vaddr);
  size_t size = ROUND_UP(phdr.p_filesz + PAGE_OFFSET(phdr.p_vaddr));
  off_t offset = phdr.p_offset - PAGE_OFFSET(phdr.p_vaddr);

  // phdr.p_vaddr = ROUND_DOWN(phdr.p_vaddr);
  // phdr.p_offset = ROUND_DOWN(phdr.p_offset);
  // phdr.p_memsz = phdr.p_memsz + (vaddr - phdr.p_vaddr);
  // NOTE:  base_address is 0 for first load segment
  if (first_time) {
	  printf("size %d \n", (int)phdr.p_filesz);
    phdr.p_vaddr += (unsigned long long)ld_so_addr;
    size = 0x27000;
  } else {
    flags |= MAP_FIXED;
  }
  if (ld_so_addr) {
    flags |= MAP_FIXED;
  }
  // FIXME:  On first load segment, we should map 0x400000 (2*phdr.p_align),
  //         and then unmap the unused portions later after all the
  //         LOAD segments are mapped.  This is what ld.so would do.
  rc2 = mmapWrapper((void *)addr, size, prot, flags, fd, offset);
  if (rc2 == MAP_FAILED) {
    DLOG(ERROR, "Failed to map memory region at %p. Error:%s\n",
         (void*)addr, strerror(errno));
    return NULL;
  }
  unsigned long startBss = (uintptr_t)base_address +
                          phdr.p_vaddr + phdr.p_filesz;
  unsigned long endBss = (uintptr_t)base_address + phdr.p_vaddr + phdr.p_memsz;
  // Required by ELF Format:
  if (phdr.p_memsz > phdr.p_filesz) {
    // This condition is true for the RW (data) segment of ld.so
    // We need to clear out the rest of memory contents, similarly to
    // what the kernel would do. See here:
    //   https://elixir.bootlin.com/linux/v4.18.11/source/fs/binfmt_elf.c#L905
    // Note that p_memsz indicates end of data (&_end)

    // First, get to the page boundary
    uintptr_t endByte = ROUND_UP(startBss);
    // Next, figure out the number of bytes we need to clear out.
    // From Bss to the end of page.
    size_t bytes = endByte - startBss;
    memset((void*)startBss, 0, bytes);
  }
  // If there's more bss that overflows to another page, map it in and
  // zero it out
  startBss  = ROUND_UP(startBss);
  endBss    = ROUND_UP(endBss);
  if (endBss > startBss) {
    void *base = (void*)startBss;
    size_t len = endBss - startBss;
    flags |= MAP_ANONYMOUS; // This should give us 0-ed out pages
    rc2 = mmapWrapper(base, len, prot, flags, -1, 0);
    if (rc2 == MAP_FAILED) {
      DLOG(ERROR, "Failed to map memory region at %p. Error:%s\n",
           (void*)startBss, strerror(errno));
      return NULL;
    }
  }
  if (first_time) {
    first_time = 0;
    base_address = (char *)rc2;
  }
  return base_address;
}
