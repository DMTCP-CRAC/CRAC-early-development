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

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "common.h"
#include "upper-half-wrappers.h"

int initialized = 0;

static void readLhInfoAddr();
extern "C" pid_t dmtcp_get_real_pid();

LowerHalfInfo_t lhInfo = {0};

void*
sbrk(intptr_t increment)
{
  static __typeof__(&sbrk) lowerHalfSbrkWrapper = (__typeof__(&sbrk)) - 1;
  if (!initialized) {
    initialize_wrappers();
  }
  if (lowerHalfSbrkWrapper == (__typeof__(&sbrk)) - 1) {
    lowerHalfSbrkWrapper = (__typeof__(&sbrk))lhInfo.lhSbrk;
  }
  // TODO: Switch fs context
  return lowerHalfSbrkWrapper(increment);
}

void*
mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
  static __typeof__(&mmap) lowerHalfMmapWrapper = (__typeof__(&mmap)) - 1;
  if (!initialized) {
    initialize_wrappers();
  }
  if (lowerHalfMmapWrapper == (__typeof__(&mmap)) - 1) {
    lowerHalfMmapWrapper = (__typeof__(&mmap))lhInfo.lhMmap;
  }
  // TODO: Switch fs context
  return lowerHalfMmapWrapper(addr, length, prot, flags, fd, offset);
}

int
munmap(void *addr, size_t length)
{
  static __typeof__(&munmap) lowerHalfMunmapWrapper = (__typeof__(&munmap)) - 1;
  if (!initialized) {
    initialize_wrappers();
  }
  if (lowerHalfMunmapWrapper == (__typeof__(&munmap)) - 1) {
    lowerHalfMunmapWrapper = (__typeof__(&munmap))lhInfo.lhMunmap;
  }
  // TODO: Switch fs context
  return lowerHalfMunmapWrapper(addr, length);
}

void
initialize_wrappers()
{
  if (!initialized) {
    readLhInfoAddr();
    initialized = 1;
  }
}

void
reset_wrappers()
{
  initialized = 0;
}

static void
readLhInfoAddr()
{
  char filename[100];
  snprintf(filename, 100, "./lhInfo_%d", dmtcp_get_real_pid());
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    printf("Could not open %s for reading.", filename);
    exit(-1);
  }
  ssize_t rc = read(fd, &lhInfo, sizeof(lhInfo));
  if (rc != (ssize_t)sizeof(lhInfo)) {
    perror("Read fewer bytes than expected from addr.bin.");
    exit(-1);
  }
//  unlink(LH_FILE_NAME);
//  close(fd);
}
