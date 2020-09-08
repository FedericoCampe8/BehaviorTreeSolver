#ifndef DLL_EXPORT_H
#define DLL_EXPORT_H

#ifdef _MSC_VER
  #define DLL_EXPORT_SYM __declspec(dllexport)
  #define DLL_EXPORT_TEMPLATE
  #define DLL_IMPORT_SYM __declspec(dllimport)
#elif __GNUC__ >= 4
  #define DLL_EXPORT_SYM __attribute__ ((visibility("default")))
  #define DLL_EXPORT_TEMPLATE __attribute__ ((visibility("default")))
  #define DLL_IMPORT_SYM __attribute__ ((visibility("default")))
#else
  #define DLL_EXPORT_SYM
  #define DLL_EXPORT_TEMPLATE
  #define DLL_IMPORT_SYM
#endif

#ifdef __cplusplus
  #define EXTERN_C extern "C"
#else
  #define EXTERN_C extern
#endif

#define EXPORT_TEMPLATE     DLL_EXPORT_TEMPLATE

/*
 * Make inline keyword work with headers shared between C and C++
 */
#if !defined(__cplusplus) && (__STDC_VERSION__ < 199901L)
#  if defined(__GNUC__)
#    define inline static __inline__
#  elif defined(_MSC_VER)
#    define inline __inline
#  endif
#endif

#endif
