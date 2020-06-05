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

#include <asm/prctl.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <errno.h>
#include <stdio.h>

#include "switch_context.h"

SwitchContext::SwitchContext(unsigned long lowerHalfFs)
{
  this->lowerHalfFs = lowerHalfFs;
  int rc = syscall(SYS_arch_prctl, ARCH_GET_FS, &this->upperHalfFs);
  if (rc < 0) {
    printf("failed to get fs: %d\n", errno);
    exit(-1);
  }
  this->jumped = 0;
  if (lowerHalfFs > 0 && lowerHalfFs != this->upperHalfFs) {
    rc = syscall(SYS_arch_prctl, ARCH_SET_FS, this->lowerHalfFs);
    if (rc < 0) {
      printf("failed to get fs: %d\n", errno);
      exit(-1);
    }
    this->jumped = 1;
  }
}

SwitchContext::~SwitchContext()
{
  if (this->jumped) {
    int rc = syscall(SYS_arch_prctl, ARCH_SET_FS, this->upperHalfFs);
    if (rc < 0) {
      printf("failed to get fs: %d\n", errno);
      exit(-1);
    }
  }
}