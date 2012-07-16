AUTOMAKE_OPTIONS = foreign

lib_LTLIBRARIES = libsifthess.la

libsifthess_la_SOURCES = imgfeatures.c kdtree.c minpq.c sift.c

libsifthess_la_LIBADD = @LTLIBOBJS@
libsifthess_la_LDFLAGS = -release @VERSION@

include_HEADERS = imgfeatures.h kdtree.h minpq.h sift.h sifthess_config.h utils.h xform.h sifthess_version.h
EXTRA_DIST = imgfeatures.h kdtree.h sift.h minpq.h utils.h xform.h sifthess_version.h
noinst_HEADERS = sifthess_config.h

if BUILD_EXAMPLES
  bin_PROGRAMS = match siftfeat dspfeat match_bench
  match_SOURCES = match.c utils.c
  siftfeat_SOURCES = siftfeat.c utils.c
  dspfeat_SOURCES = dspfeat.c utils.c
  match_bench_SOURCES = match_bench.c utils.c

  match_LDADD = $(GTK_ENGINES_2_LIBS) $(ADD_OPENCV_LIBS) libsifthess.la
  siftfeat_LDADD = $(GTK_ENGINES_2_LIBS) $(ADD_OPENCV_LIBS) libsifthess.la
  dspfeat_LDADD = $(GTK_ENGINES_2_LIBS) $(ADD_OPENCV_LIBS) libsifthess.la
  match_bench_LDADD = $(GTK_ENGINES_2_LIBS) $(ADD_OPENCV_LIBS) libsifthess.la

  match_CFLAGS = $(GTK_ENGINES_2_CFLAGS)
  siftfeat_CFLAGS= $(GTK_ENGINES_2_CFLAGS)
  dspfeat_CFLAGS = $(GTK_ENGINES_2_CFLAGS)
  match_bench_CFLAGS = $(GTK_ENGINES_2_CFLAGS)
endif

