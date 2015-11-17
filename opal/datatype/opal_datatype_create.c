/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include <stddef.h>

#include "opal/constants.h"
#include "opal/datatype/opal_datatype.h"
#include "opal/datatype/opal_datatype_internal.h"
#include "limits.h"
#include "opal/prefetch.h"
#if OPAL_CUDA_SUPPORT
#include "opal/datatype/opal_convertor.h"
#include "opal/datatype/opal_datatype_cuda.h"
#endif /* OPAL_CUDA_SUPPORT */ 

static void opal_datatype_construct( opal_datatype_t* pData )
{
    int i;

    pData->size               = 0;
    pData->flags              = OPAL_DATATYPE_FLAG_CONTIGUOUS;
    pData->id                 = 0;
    pData->bdt_used           = 0;
    pData->size               = 0;
    pData->true_lb            = LONG_MAX;
    pData->true_ub            = LONG_MIN;
    pData->lb                 = LONG_MAX;
    pData->ub                 = LONG_MIN;
    pData->align              = 1;
    pData->nbElems            = 0;
    memset(pData->name, 0, OPAL_MAX_OBJECT_NAME);

    pData->desc.desc          = NULL;
    pData->desc.length        = 0;
    pData->desc.used          = 0;

    pData->opt_desc.desc      = NULL;
    pData->opt_desc.length    = 0;
    pData->opt_desc.used      = 0;

    pData->cached_iovec       = NULL;
    pData->cached_iovec_count = 0;
    
#if OPAL_CUDA_SUPPORT
    pData->cached_cuda_iov = NULL;
#endif /* OPAL_CUDA_SUPPORT */

    for( i = 0; i < OPAL_DATATYPE_MAX_SUPPORTED; i++ )
        pData->btypes[i]      = 0;
}

static void opal_datatype_destruct( opal_datatype_t* datatype )
{
    if (!opal_datatype_is_predefined(datatype)) {
        if( datatype->desc.desc != NULL ) {
            free( datatype->desc.desc );
            datatype->desc.length = 0;
            datatype->desc.used   = 0;
        }
    }
    if( datatype->opt_desc.desc != NULL ) {
        if( datatype->opt_desc.desc != datatype->desc.desc )
            free( datatype->opt_desc.desc );
        datatype->opt_desc.length = 0;
        datatype->opt_desc.used   = 0;
        datatype->opt_desc.desc   = NULL;
    }
    /**
     * As the default description and the optimized description can point to the
     * same memory location we should keep the default location pointer until we
     * know what we should do with the optimized description.
     */
    datatype->desc.desc   = NULL;

    /* make sure the name is set to empty */
    datatype->name[0] = '\0';

    if( NULL != datatype->cached_iovec ) {
        free(datatype->cached_iovec);
        datatype->cached_iovec = NULL;
    }
    
#if OPAL_CUDA_SUPPORT   
    /* free cuda iov */
    if (opal_datatype_cuda_kernel_support == 1 && datatype->cached_cuda_iov != NULL) {
        opal_cached_cuda_iov_fini((void*)datatype->cached_cuda_iov);
        datatype->cached_cuda_iov = NULL;
    }
#endif /* OPAL_CUDA_SUPPORT */
}

OBJ_CLASS_INSTANCE(opal_datatype_t, opal_object_t, opal_datatype_construct, opal_datatype_destruct);

opal_datatype_t* opal_datatype_create( int32_t expectedSize )
{
    opal_datatype_t* datatype = (opal_datatype_t*)OBJ_NEW(opal_datatype_t);

    if( expectedSize == -1 ) expectedSize = DT_INCREASE_STACK;
    datatype->desc.length = expectedSize + 1;  /* one for the fake elem at the end */
    datatype->desc.used   = 0;
    datatype->desc.desc   = (dt_elem_desc_t*)calloc(datatype->desc.length, sizeof(dt_elem_desc_t));
    /* BEWARE: an upper-layer configured with OPAL_MAX_OBJECT_NAME different than the OPAL-layer will not work! */
    memset( datatype->name, 0, OPAL_MAX_OBJECT_NAME );
    return datatype;
}

int32_t opal_datatype_create_desc( opal_datatype_t * datatype, int32_t expectedSize )
{
    if( expectedSize == -1 )
        expectedSize = DT_INCREASE_STACK;
    datatype->desc.length = expectedSize + 1;  /* one for the fake elem at the end */
    datatype->desc.used   = 0;
    datatype->desc.desc   = (dt_elem_desc_t*)calloc(datatype->desc.length, sizeof(dt_elem_desc_t));
    if (NULL == datatype->desc.desc)
        return OPAL_ERR_OUT_OF_RESOURCE;
    return OPAL_SUCCESS;
}
