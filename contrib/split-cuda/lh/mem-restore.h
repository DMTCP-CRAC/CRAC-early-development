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

#ifndef MEM_RESTORE_H
#define MEM_RESTORE_H

#include "tlsutil.h"
#include "mtcp_header.h"
#include "procmapsutils.h"

#ifdef __cplusplus
  extern "C" {
#endif
    void restoreCheckpointImg(int );
    void returnTodmtcp();
    pid_t getUhPid();
#ifdef __cplusplus
  }
#endif
static int restoreMemory(int );
static int restoreMemoryRegion(int , Area* );
static int readMtcpHeader(int );
extern void restore_libc(ThreadTLSInfo *tlsInfo,
                int tls_pid_offset,
                int tls_tid_offset,
                MYINFO_GS_T myinfo_gs);
#endif // ifndef MEM_RESTORE_H
