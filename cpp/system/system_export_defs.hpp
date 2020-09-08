#pragma once

#include "dll_export.hpp"

/* This header is being included by files inside this module */

#define SYS_EXPORT_CLASS     DLL_EXPORT_SYM
#define SYS_EXPORT_STRUCT    DLL_EXPORT_SYM
#define SYS_EXPORT_FRIEND    DLL_EXPORT_SYM
#define SYS_EXPORT_FCN       DLL_EXPORT_SYM
#define SYS_EXPORT_VAR       DLL_EXPORT_SYM
#define SYS_EXPORT_TEMPLATE  DLL_EXPORT_TEMPLATE
#define SYS_EXPORT_EXTERN_C  extern "C" DLL_EXPORT_SYM
