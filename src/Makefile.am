AUTOMAKE_OPTIONS = foreign

lib_LTLIBRARIES = libsifthess.la

libsifthess_la_SOURCES = imgfeatures.c kdtree.c minpq.c sift.c

libsifthess_la_LIBADD = @LTLIBOBJS@
libsifthess_la_LDFLAGS = -release @VERSION@

if BUILD_EXAMPLES
  bin_PROGRAMS = match siftfeat dspfeat
  match_SOURCES = match.c utils.c
  siftfeat_SOURCES = siftfeat.c utils.c
  dspfeat_SOURCES = dspfeat.c utils.c

  match_LDADD = $(GTK_ENGINES_2_LIBS) $(ADD_OPENCV_LIBS) libsifthess.la
  siftfeat_LDADD = $(GTK_ENGINES_2_LIBS) $(ADD_OPENCV_LIBS) libsifthess.la
  dspfeat_LDADD = $(GTK_ENGINES_2_LIBS) $(ADD_OPENCV_LIBS) libsifthess.la

  match_CFLAGS = $(GTK_ENGINES_2_CFLAGS)
  siftfeat_CFLAGS= $(GTK_ENGINES_2_CFLAGS)
  dspfeat_CFLAGS = $(GTK_ENGINES_2_CFLAGS)
endif

