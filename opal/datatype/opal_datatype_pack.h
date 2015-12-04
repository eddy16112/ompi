/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2009 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2011      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_DATATYPE_PACK_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_PACK_H_HAS_BEEN_INCLUDED

#include "opal_config.h"

#include "opal/datatype/opal_datatype_ht.h"
#include "opal/sys/atomic.h"

#include <stddef.h>

#if !defined(CHECKSUM) && OPAL_CUDA_SUPPORT
/* Make use of existing macro to do CUDA style memcpy */
#undef MEMCPY_CSUM
#define MEMCPY_CSUM( DST, SRC, BLENGTH, CONVERTOR ) \
    CONVERTOR->cbmemcpy( (DST), (SRC), (BLENGTH), (CONVERTOR) )
#endif

static inline void pack_predefined_data( opal_convertor_t* CONVERTOR,
                                         const dt_elem_desc_t* ELEM,
                                         uint32_t* COUNT,
                                         unsigned char** SOURCE,
                                         unsigned char** DESTINATION,
                                         size_t* SPACE )
{
    uint32_t _copy_count = *(COUNT);
    size_t _copy_blength;
    const ddt_elem_desc_t* _elem = &((ELEM)->elem);
    unsigned char* _source = (*SOURCE) + _elem->disp;

    _copy_blength = opal_datatype_basicDatatypes[_elem->common.type]->size;
    if( (_copy_count * _copy_blength) > *(SPACE) ) {
        _copy_count = (uint32_t)(*(SPACE) / _copy_blength);
        if( 0 == _copy_count ) return;  /* nothing to do */
    }

    if( (OPAL_PTRDIFF_TYPE)_copy_blength == _elem->extent ) {
        _copy_blength *= _copy_count;
        /* the extent and the size of the basic datatype are equal */
        OPAL_DATATYPE_SAFEGUARD_POINTER( _source, _copy_blength, (CONVERTOR)->pBaseBuf,
                                    (CONVERTOR)->pDesc, (CONVERTOR)->count );
        DO_DEBUG( opal_output( 0, "pack 1. memcpy( %p, %p, %lu ) => space %lu\n",
                               *(DESTINATION), _source, (unsigned long)_copy_blength, (unsigned long)(*(SPACE)) ); );
        MEMCPY_CSUM( *(DESTINATION), _source, _copy_blength, (CONVERTOR) );
        _source        += _copy_blength;
        *(DESTINATION) += _copy_blength;
    } else {
        uint32_t _i;
        for( _i = 0; _i < _copy_count; _i++ ) {
            OPAL_DATATYPE_SAFEGUARD_POINTER( _source, _copy_blength, (CONVERTOR)->pBaseBuf,
                                        (CONVERTOR)->pDesc, (CONVERTOR)->count );
            DO_DEBUG( opal_output( 0, "pack 2. memcpy( %p, %p, %lu ) => space %lu\n",
                                   *(DESTINATION), _source, (unsigned long)_copy_blength, (unsigned long)(*(SPACE) - (_i * _copy_blength)) ); );
            MEMCPY_CSUM( *(DESTINATION), _source, _copy_blength, (CONVERTOR) );
            *(DESTINATION) += _copy_blength;
            _source        += _elem->extent;
        }
        _copy_blength *= _copy_count;
    }
    *(SOURCE)  = _source - _elem->disp;
    *(SPACE)  -= _copy_blength;
    *(COUNT)  -= _copy_count;
}

static inline void pack_contiguous_loop( opal_convertor_t* CONVERTOR,
                                         const dt_elem_desc_t* ELEM,
                                         uint32_t* COUNT,
                                         unsigned char** SOURCE,
                                         unsigned char** DESTINATION,
                                         size_t* SPACE )
{
    const ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    const ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _source = (*SOURCE) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    int32_t _i;
#if defined (OPAL_DATATYPE_PARALLEL)
    datatype_ht_desc_t* ht_desc;
    datatype_ht_pool_t* ht_pool;
    contiguous_loop_ht_data_t* ht_data;
    unsigned char* _destination;
    int32_t begi, endi, chunk_size, _j;
#endif /* defined (OPAL_DATATYPE_PARALLEL) */

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);
#if !defined(OPAL_DATATYPE_PARALLEL)
    for( _i = 0; _i < _copy_loops; _i++ ) {
        OPAL_DATATYPE_SAFEGUARD_POINTER( _source, _end_loop->size, (CONVERTOR)->pBaseBuf,
                                    (CONVERTOR)->pDesc, (CONVERTOR)->count );
        DO_DEBUG( opal_output( 0, "pack 3. memcpy( %p, %p, %lu ) => space %lu\n",
                               *(DESTINATION), _source, (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _i * _end_loop->size) ); );
        MEMCPY_CSUM( *(DESTINATION), _source, _end_loop->size, (CONVERTOR) );
        *(DESTINATION) += _end_loop->size;
        _source        += _loop->extent;
    }
#else
    
    if (_copy_loops > 0) {
        cl_ht_data.source = _source;
        cl_ht_data.destination = *(DESTINATION);
        cl_ht_data.copy_loops = _copy_loops;
        cl_ht_data.loop = _loop;
        cl_ht_data.end_loop = _end_loop;
        cl_ht_data.CONVERTOR = CONVERTOR;
        cl_ht_data.SPACE = SPACE;
        cl_ht_data.callback_source = _source;
        cl_ht_data.callback_destination = *(DESTINATION);
    
        for( _i = 0; _i < (int32_t)dt_ht_pool.num_ht; _i++ ) {
            dt_ht_desc[_i].ht_data = (datatype_ht_data_t*)&cl_ht_data;
            dt_ht_desc[_i].task = opal_datatype_ht_pack_contiguous_loop;
        }
        
        ht_desc = &dt_ht_desc[0];
        ht_pool = ht_desc->ht_pool;
        ht_data = (contiguous_loop_ht_data_t*)ht_desc->ht_data;
        _destination = *(DESTINATION);
        
        opal_atomic_mb();
        
        ht_pool->loop_unfini = _copy_loops;
        ht_pool->ht_fini = 0;
        ht_pool->ht_wake = 1;
        ht_desc->num_task_done = 0;
//        printf("before bcast src %p, dst %p, copy_loops %d \n", _source, *(DESTINATION), _copy_loops);
        // printf("tid %d, src %p [base %p], dst %p [base %p], cl %d extent %ld size %lu loop_ct %d\n",
        //        ht_desc->thread_id, _source, cl_ht_data.source, _destination,
        //        cl_ht_data.destination, _copy_loops, _loop->extent, _end_loop->size, ht_pool->loop_unfini);
        
        if (ht_pool->num_ht == 1) {
            chunk_size = _copy_loops;
        } else {
            pthread_mutex_lock(&(dt_ht_pool.ht_lock));
            pthread_cond_broadcast(&(dt_ht_pool.q_notempty)); 
            pthread_mutex_unlock(&(dt_ht_pool.ht_lock));
            chunk_size = _copy_loops / DATATYPE_CUT; 
        }
        
        while (ht_pool->loop_unfini > 0) {
            endi = opal_atomic_sub_32(&(ht_pool->loop_unfini), chunk_size);
            begi = endi + chunk_size - 1;
  //        printf("unfini %d， chunk_size %d, beg %d, end %d\n", ht_pool->loop_unfini, chunk_size, begi, endi);
            for (_i = begi; _i >= endi; _i--) {
                if (_i < 0) {
                    break;
                }
                _j = _copy_loops - (_i + 1);
                _source = ht_data->source + _j * _loop->extent;
                _destination = ht_data->destination + _j * _end_loop->size;

                OPAL_DATATYPE_SAFEGUARD_POINTER( _source, _loop->extent, (CONVERTOR)->pBaseBuf,
                                                (CONVERTOR)->pDesc, (CONVERTOR)->count );
                // OPAL_DATATYPE_SAFEGUARD_POINTER( _destination, _end_loop->size, ht_data->destination,
                //                                 (CONVERTOR)->pDesc, (CONVERTOR)->count );
                DO_DEBUG( opal_output( 0, "pack 3. memcpy( %p, %p, %lu ) => space %lu\n",
                                       _destination, _source, (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _j * _end_loop->size) ); );
                // printf ("tid %d, pack 3. memcpy( %p, %p, %lu ) => space %lu, i %d\n",
                //         ht_desc->thread_id, _destination, _source, (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _j * _end_loop->size), _j );
                MEMCPY_CSUM( _destination, _source, _end_loop->size, (CONVERTOR) );
                ht_desc->num_task_done ++;
            }
            chunk_size = ht_pool->loop_unfini / DATATYPE_CUT;
            if (chunk_size < DATATYPE_CHUNK_SIZE_LIMIT) {
                chunk_size = DATATYPE_CHUNK_SIZE_LIMIT;
            }
        }
        opal_atomic_add_32(&(ht_pool->ht_fini), 1);

        while ( ht_pool->ht_fini < ht_pool->ht_wake ) {
        }
        //printf("total %d \n", ht_pool->loop_fini+ht_pool->loop_unfini);
        _source = ht_data->source  + _copy_loops * _loop->extent;
        *(DESTINATION) = ht_data->destination  + _copy_loops * _end_loop->size;
//      printf("received src %p, dst %p\n", _source, *(DESTINATION));
        printf("pthread %d, task %d, total %d\n", ht_desc->thread_id, ht_desc->num_task_done, _copy_loops);
        assert(_source != NULL);
        assert(*(DESTINATION) != NULL);
        
        for( _i = 0; _i < (int32_t)ht_pool->num_ht; _i++ ) {
            dt_ht_desc[_i].ht_data = NULL;
     //       dt_ht_desc[_i].task = NULL;
        }
    }
    
#endif
    *(SOURCE) = _source - _end_loop->first_elem_disp;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
}

#define PACK_PREDEFINED_DATATYPE( CONVERTOR,    /* the convertor */                       \
                                  ELEM,         /* the basic element to be packed */      \
                                  COUNT,        /* the number of elements */              \
                                  SOURCE,       /* the source pointer (char*) */          \
                                  DESTINATION,  /* the destination pointer (char*) */     \
                                  SPACE )       /* the space in the destination buffer */ \
pack_predefined_data( (CONVERTOR), (ELEM), &(COUNT), &(SOURCE), &(DESTINATION), &(SPACE) )

#define PACK_CONTIGUOUS_LOOP( CONVERTOR, ELEM, COUNT, SOURCE, DESTINATION, SPACE ) \
    pack_contiguous_loop( (CONVERTOR), (ELEM), &(COUNT), &(SOURCE), &(DESTINATION), &(SPACE) )

#endif  /* OPAL_DATATYPE_PACK_H_HAS_BEEN_INCLUDED */
