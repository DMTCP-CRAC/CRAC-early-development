#include <assert.h>

#include "logging.h" // for DLOG
#include "lh_pluginmanager.h"
#include "lh_plugininfo.h"

static void
printUsage()
{
  DLOG(ERROR, "Usage: UH_PRELOAD=/path/to/libupperhalfwrappers.so "
          "TARGET_LD=/path/to/ld.so ./kernel-loader "
          "<target-application> [application arguments ...]\n");
}

static void
printRestartUsage()
{
  DLOG(ERROR, "Usage: ./kernel-loader --restore /path/to/ckpt.img\n");
}

int
main(int argc, char *argv[], char **environ)
{
  LowerHalf::PluginManager::eventHook(LH_EVENT_INIT, NULL);
  LowerHalf::PluginManager::eventHook(LH_EVENT_PRECHECKPOINT, NULL);
  LowerHalf::PluginManager::eventHook(LH_EVENT_RESUME, NULL);
  return 0;
}
