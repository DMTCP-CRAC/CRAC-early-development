#include "lh_pluginmanager.h"
#include "lh_plugininfo.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ENV_VAR_CUDA_PLUGIN "DMTCP_CUDA_PLUGIN"
#define ENV_VAR_MPI_PLUGIN "DMTCP_MPI_PLUGIN"

static LowerHalf::PluginManager *pluginManager = NULL;

extern "C" void
lh_register_plugin(LhPluginDescriptor_t descr)
{
  assert(pluginManager != NULL);

  pluginManager->registerPlugin(descr);
}

namespace LowerHalf
{
LhPluginDescriptor_t lh_cuda_PluginDescr();
LhPluginDescriptor_t lh_mpi_PluginDescr();

void
PluginManager::initialize()
{
  if (pluginManager == NULL) {
    pluginManager = new PluginManager();
    // Now initialize plugins.
    lh_initialize_plugin();
  }
}

PluginManager::PluginManager()
{}

void
PluginManager::registerPlugin(LhPluginDescriptor_t descr)
{
  PluginInfo *info = new PluginInfo(descr);

  pluginInfos.push_back(info);
}

extern "C" void
lh_initialize_plugin()
{
  // register the "in-built" plugins.
  const char *ptr = getenv(ENV_VAR_CUDA_PLUGIN);
  if (ptr != NULL && strcmp(ptr, "1") == 0) {
    lh_register_plugin(lh_cuda_PluginDescr());
  }
  ptr = getenv(ENV_VAR_MPI_PLUGIN);
  if (ptr != NULL && strcmp(ptr, "1") == 0) {
    lh_register_plugin(lh_mpi_PluginDescr());
  }
}


void
PluginManager::eventHook(LhEvent_t event, LhEventData_t *data)
{
  PluginManager::initialize();

  switch (event) {
  // The following events are processed in the order of plugin registration.
  case LH_EVENT_INIT:
  case LH_EVENT_RUNNING:
  case LH_EVENT_PRE_EXEC:
  case LH_EVENT_POST_EXEC:
  case LH_EVENT_ATFORK_PREPARE:
  case LH_EVENT_VFORK_PREPARE:
  case LH_EVENT_PTHREAD_START:
  case LH_EVENT_OPEN_FD:
  case LH_EVENT_REOPEN_FD:
  case LH_EVENT_CLOSE_FD:
  case LH_EVENT_DUP_FD:
  case LH_EVENT_VIRTUAL_TO_REAL_PATH:
  case LH_EVENT_PRESUSPEND:
  case LH_EVENT_PRECHECKPOINT:

    // The plugins can be thought of as implementing a layered software
    // architecture.  All of the events here occur before writing the checkpoint
    // file.  The plugins are invoked for these events in the natural order.
    // For the resume/restart events below, the plugins are invoked
    // in _reverse_ order.  This is required to support layered software.
    // For an analogous case, see 'man pthread_atfork' with the handlers:
    // (i) prepare, (ii) parent, and (iii) child.
    // Those are analogous to our events for:
    // (i) pre-checkpoint, (ii) resume event, and (iii) restart; respectively.
    for (size_t i = 0; i < pluginManager->pluginInfos.size(); i++) {
      if (pluginManager->pluginInfos[i]->event_hook) {
        pluginManager->pluginInfos[i]->event_hook(event, data);
      }
    }
    break;

  // The following events are processed in reverse order.
  case LH_EVENT_EXIT:
  case LH_EVENT_PTHREAD_EXIT:
  case LH_EVENT_PTHREAD_RETURN:
  case LH_EVENT_ATFORK_PARENT:
  case LH_EVENT_ATFORK_CHILD:
  case LH_EVENT_ATFORK_FAILED:
  case LH_EVENT_VFORK_PARENT:
  case LH_EVENT_VFORK_CHILD:
  case LH_EVENT_VFORK_FAILED:
  case LH_EVENT_REAL_TO_VIRTUAL_PATH:
  case LH_EVENT_RESUME:
  case LH_EVENT_RESTART:
  case LH_EVENT_THREAD_RESUME:
    // The plugins are invoked in _reverse_ order during resume/restart.  This
    // is required to support layered software.  See the related comment, above.
    for (int i = pluginManager->pluginInfos.size() - 1; i >= 0; i--) {
      if (pluginManager->pluginInfos[i]->event_hook) {
        pluginManager->pluginInfos[i]->event_hook(event, data);
      }
    }
    break;
  default:
    fprintf(stderr, "Lower-half event not reachable!\n");
  }
}
} // namespace LowerHalf {
