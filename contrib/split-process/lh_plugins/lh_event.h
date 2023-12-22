#ifndef __LH_EVENT_H__
#define __LH_EVENT_H__

#include <sys/types.h>

// NOTE: we may not need all the events inherited from DMTCP. However, we can
// keep them for future.
typedef enum eLHEvent {
  LH_EVENT_INIT,
  LH_EVENT_EXIT,

  LH_EVENT_PRE_EXEC,
  LH_EVENT_POST_EXEC,

  LH_EVENT_ATFORK_PREPARE,
  LH_EVENT_ATFORK_PARENT,
  LH_EVENT_ATFORK_CHILD,
  LH_EVENT_ATFORK_FAILED,

  LH_EVENT_VFORK_PREPARE,
  LH_EVENT_VFORK_PARENT,
  LH_EVENT_VFORK_CHILD,
  LH_EVENT_VFORK_FAILED,

  LH_EVENT_PTHREAD_START,
  LH_EVENT_PTHREAD_EXIT,
  LH_EVENT_PTHREAD_RETURN,

  LH_EVENT_PRESUSPEND,
  LH_EVENT_PRECHECKPOINT,
  LH_EVENT_RESUME,
  LH_EVENT_RESTART,
  LH_EVENT_RUNNING,
  LH_EVENT_THREAD_RESUME,

  LH_EVENT_OPEN_FD,
  LH_EVENT_REOPEN_FD,
  LH_EVENT_CLOSE_FD,
  LH_EVENT_DUP_FD,

  LH_EVENT_VIRTUAL_TO_REAL_PATH,
  LH_EVENT_REAL_TO_VIRTUAL_PATH,

  nLhEvents
} LhEvent_t;

typedef union _LhEventData_t {
  struct {
    int serializationFd;
    char *filename;
    size_t maxArgs;
    const char **argv;
    size_t maxEnv;
    const char **envp;
  } preExec;

  struct {
    int serializationFd;
  } postExec;

  struct {
    int isRestart;
  } resumeUserThreadInfo, nameserviceInfo;

  struct {
    int fd;
    const char *path;
    int flags;
    mode_t mode;
  } openFd;

  struct {
    int fd;
    const char *path;
    int flags;
  } reopenFd;

  struct {
    int fd;
  } closeFd;

  struct {
    int oldFd;
    int newFd;
  } dupFd;

  struct {
    char *path;
  } realToVirtualPath, virtualToRealPath;
} LhEventData_t;

#endif // __LH_EVENT_H__
