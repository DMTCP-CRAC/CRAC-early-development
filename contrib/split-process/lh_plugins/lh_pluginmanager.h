/****************************************************************************
 *   Copyright (C) 2006-2013 by Jason Ansel, Kapil Arya, and Gene Cooperman *
 *   jansel@csail.mit.edu, kapil@ccs.neu.edu, gene@ccs.neu.edu              *
 *                                                                          *
 *   Copyright (C) 2023 by Twinkle Jain                                     *
 *   jain.t@northeastern.edu                                                *
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

#ifndef __LH_PLUGINMANAGER_H__
#define __LH_PLUGINMANAGER_H__

#include "lh_plugininfo.h"
#include <vector>

namespace LowerHalf
{
class PluginManager
{
  public:
    PluginManager();

    void registerPlugin(LhPluginDescriptor_t descr);

    static void initialize();
    static void processPreSuspendBarriers();
    static void processCkptBarriers();
    static void processResumeBarriers();
    static void processRestartBarriers();
    static void eventHook(LhEvent_t event, LhEventData_t *data = NULL);

  private:
    std::vector<PluginInfo *>pluginInfos;
};
}
#endif // ifndef __LH_PLUGINMANAGER_H__
