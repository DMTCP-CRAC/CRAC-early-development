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

#ifndef UPPER_HALF_CUDA_WRAPPERS_H
#define UPPER_HALF_CUDA_WRAPPERS_H

#include "dmtcp.h"

#define REAL_FNC(fnc) \
  ({ fnc##_t fnc##Fnc = (fnc##_t) -1; \
  if (!initialized) { \
    initialize_wrappers(); \
  } \
  if (fnc##Fnc == (fnc##_t) -1) { \
    LhDlsym_t dlsymFptr = (LhDlsym_t)lhInfo.lhDlsym; \
    fnc##Fnc = (fnc##_t)dlsymFptr(Cuda_Fnc_##fnc); \
  } \
  fnc##Fnc; })

#define USER_DEFINED_WRAPPER(rettype, fnc, args) \
  typedef rettype (*fnc##_t)args;              \
  EXTERNC rettype fnc args

#define DECLARE_WRAPPER(rettype, fnc, args) \
  extern rettype fnc args __attribute__((weak));
void **global_fatCubinHandle=NULL;
#endif // ifndef UPPER_HALF_CUDA_WRAPPERS_H
