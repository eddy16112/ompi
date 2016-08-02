#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2010 Cisco Systems, Inc.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

AM_CPPFLAGS = $(LTDLINCL)

# main library setup
noinst_LTLIBRARIES = libmca_coll.la
libmca_coll_la_SOURCES =

# local files
headers = coll.h
libmca_coll_la_SOURCES += $(headers)

# Conditionally install the header files
if WANT_INSTALL_HEADERS
ompidir = $(ompiincludedir)/$(subdir)
nobase_ompi_HEADERS = $(headers)
endif

libmca_coll_la_LIBADD = \
    $(OMPI_TOP_BUILDDIR)/opal/mca/common/cuda/lib@OPAL_LIB_PREFIX@mca_common_cuda.la

include base/Makefile.am

distclean-local:
	rm -f base/static-components.h
