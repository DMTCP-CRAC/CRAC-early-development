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

#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include "mmap-wrapper.h"
// Returns pointer to argc, given a pointer to end of stack
static inline void*
GET_ARGC_ADDR(const void* stackEnd)
{
  return (void*)((uintptr_t)(stackEnd) + sizeof(uintptr_t));
}

// Returns pointer to argv[0], given a pointer to end of stack
static inline void*
GET_ARGV_ADDR(const void* stackEnd)
{
  return (void*)((unsigned long)(stackEnd) + 2 * sizeof(uintptr_t));
}

// Returns pointer to env[0], given a pointer to end of stack
static inline void*
GET_ENV_ADDR(char **argv, int argc)
{
  return (void*)&argv[argc + 1];
}

// Returns a pointer to aux vector, given a pointer to the environ vector
// on the stack
static inline ElfW(auxv_t)*
GET_AUXV_ADDR(const char **env)
{
  ElfW(auxv_t) *auxvec;
  const char **evp = env;
  while (*evp++ != NULL);
  auxvec = (ElfW(auxv_t) *) evp;
  return auxvec;
}
#ifdef __cplusplus
extern "C" {
#endif
  void runRtld();
  void* sbrkWrapper(intptr_t );
  void setUhBrk(void *);
  void copy_lower_half_data();
#ifdef __cplusplus
}
#endif

#endif // ifndef KERNEL_LOADER_H
