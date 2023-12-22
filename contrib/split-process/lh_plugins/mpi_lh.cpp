/* FILE: mpi_lh.cpp
 * AUTHOR: Twinkle Jain
 * EMAIL: jain.t@northeastern.edu
 * Copyright (C) 2024 Twinkle Jain
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "lh_plugininfo.h"

namespace LowerHalf
{

static void
init_msg()
{
  printf("MPI: in init_msg\n");
  fflush(stdout);
}


static void
checkpoint()
{
  printf("MPI: in checkpoint\n");
  fflush(stdout);
}

static void
resume()
{
  printf("MPI: in resume\n");
  fflush(stdout);
}

static void
mpi_EventHook(LhEvent_t event, LhEventData_t *data)
{
  switch (event) {
  case LH_EVENT_INIT:
    init_msg();
    break;
  case LH_EVENT_PRECHECKPOINT:
    checkpoint();
    break;

  case LH_EVENT_RESUME:
    resume();
    break;

  default:
    break;
  }
}

static LhPluginDescriptor_t mpiPlugin = {
  LH_PLUGIN_API_VERSION,
  PACKAGE_VERSION,
  "MPI",
  "LH",
  "jain.t@northeastern.edu",
  "mpi plugin",
  mpi_EventHook
};


LhPluginDescriptor_t
lh_mpi_PluginDescr()
{
  return mpiPlugin;
}
}
