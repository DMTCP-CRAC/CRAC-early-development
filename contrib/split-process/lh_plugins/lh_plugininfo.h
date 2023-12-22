/****************************************************************************
 *   Copyright (C) 2006-2013 by Jason Ansel, Kapil Arya, and Gene Cooperman *
 *   jansel@csail.mit.edu, kapil@ccs.neu.edu, gene@ccs.neu.edu              *
 *                                                                          *
 *   Copyright (C) 2023 by Twinkle Jain                                     *
 *   jain.t@northeastern.edu                                                *
 *   This file is part of DMTCP.                                            *
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

#ifndef __LH_PLUGININFO_H__
#define __LH_PLUGININFO_H__

#include "lh_event.h"

#define LH_PLUGIN_API_VERSION "1"
#define PACKAGE_VERSION "2.0.0"

typedef struct {
  const char *pluginApiVersion;
  const char *SplitProcessVersion;

  const char *pluginName;
  const char *authorName;
  const char *authorEmail;
  const char *description;

  void (*event_hook)(const LhEvent_t event, LhEventData_t *data);
} LhPluginDescriptor_t;

// Internal usage only. Shouldn't be used directly by the plugin. Use
// LH_DECL_PLUGIN instead.
extern "C" void lh_initialize_plugin(void) __attribute((weak));
extern "C" void lh_register_plugin(LhPluginDescriptor_t) __attribute((weak));

#define LH_DECL_PLUGIN(descr)                         \
  EXTERNC void lh_initialize_plugin()                 \
  {                                                   \
    lh_register_plugin(descr);                        \
    void (*fn)() = NEXT_FNC(lh_initialize_plugin);    \
    if (fn != NULL) {                                 \
      (*fn)();                                        \
    }                                                 \
  }

namespace LowerHalf
{
class PluginInfo
{
  public:
    PluginInfo(const LhPluginDescriptor_t &descr)
      : pluginName(descr.pluginName),
        authorName(descr.authorName),
        authorEmail(descr.authorEmail),
        description(descr.description),
        event_hook(descr.event_hook)
    {}

    const char *pluginName;
    const char *authorName;
    const char *authorEmail;
    const char *description;
    void(*const event_hook)(const LhEvent_t event, LhEventData_t * data);
};
}
#endif // ifndef __LH_PLUGININFO_H__
