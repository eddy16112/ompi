/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2013 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

//#if defined (OPAL_DATATYPE_PARALLEL)

#include "opal_config.h"

#include <stddef.h>
#include <unistd.h>

#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#if OPAL_ENABLE_DEBUG
#include "opal/util/output.h"

#define DO_DEBUG(INST)  if( opal_pack_debug ) { INST }
#else
#define DO_DEBUG(INST)
#endif  /* OPAL_ENABLE_DEBUG */

#include "opal/datatype/opal_datatype_checksum.h"
#include "opal/datatype/opal_datatype_ht.h"
#include "opal/datatype/opal_datatype_prototypes.h"

#if defined(CHECKSUM)
#define opal_datatype_ht_pack_contiguous_loop_function opal_datatype_ht_pack_contiguous_loop_checksum
#else
#define opal_datatype_ht_pack_contiguous_loop_function opal_datatype_ht_pack_contiguous_loop
#endif /* CHECKSUM */

datatype_ht_desc_t dt_ht_desc[NUM_HT];
datatype_ht_pool_t dt_ht_pool;
contiguous_loop_ht_data_t cl_ht_data;

static hwloc_topology_t dt_topology;

int32_t opal_datatype_bind_thread(datatype_ht_desc_t* ht_desc);

int32_t opal_datatype_bind_thread(datatype_ht_desc_t* ht_desc)
{
    char *str;
    hwloc_set_thread_cpubind(dt_topology, (hwloc_thread_t)ht_desc->TID, ht_desc->cpuset, HWLOC_CPUBIND_THREAD);
    hwloc_bitmap_asprintf(&str, ht_desc->cpuset);
    printf("thread cpuset %s\n", str);
    return OPAL_SUCCESS;
}

int32_t opal_datatype_ht_init(void)
{
    uint32_t _i, rc;
    char *str;
    hwloc_obj_t obj, pu_obj;
    int depth;
    
    dt_ht_pool.num_ht = NUM_HT;
    dt_ht_pool.loop_unfini = 0;
    dt_ht_pool.ht_fini = 0;
    
    cl_ht_data.source = NULL;
    cl_ht_data.destination = NULL;
    cl_ht_data.copy_loops = 0;
    cl_ht_data.loop = 0;
    cl_ht_data.end_loop = 0;
    cl_ht_data.CONVERTOR = NULL;
    cl_ht_data.SPACE = NULL;
    cl_ht_data.callback_source = NULL;
    cl_ht_data.callback_destination = NULL;
    
    if (datatype_ht_init == 0) {
        hwloc_topology_init(&dt_topology);
        hwloc_topology_load(dt_topology);
        if (hwloc_get_nbobjs_by_type(dt_topology, HWLOC_OBJ_PU) > 0) {
            printf("Hyperthreading is supported, %d .\n", hwloc_get_nbobjs_by_type(dt_topology, HWLOC_OBJ_PU));
        }

        // dt_ht_desc[0].cpuset = hwloc_bitmap_alloc();
        // hwloc_get_thread_cpubind(dt_topology, (hwloc_thread_t)TID, dt_ht_desc[0].cpuset, HWLOC_CPUBIND_THREAD);
        // hwloc_bitmap_asprintf(&str, dt_ht_desc[0].cpuset);
        // printf("main thread cpuset %s\n", str);
        
        /* verify the number of hyperthread per core */
        depth = hwloc_get_type_or_below_depth(dt_topology, HWLOC_OBJ_CORE);
        obj = hwloc_get_obj_by_depth(dt_topology, depth, 0);
        if (obj != NULL) {
            dt_ht_pool.num_ht = hwloc_get_nbobjs_inside_cpuset_by_type(dt_topology, obj->cpuset, HWLOC_OBJ_PU);           
            if (dt_ht_pool.num_ht < NUM_HT) {
                printf ("Warning: NUM_HT is %d, which is more than the number of hyperthread per core %d\n", NUM_HT, dt_ht_pool.num_ht);
            } else {
                dt_ht_pool.num_ht = NUM_HT;
            }
            printf("number of PU per core: %d\n", dt_ht_pool.num_ht );
        } else {
            printf("ERROR: hwloc can not find core 0\n");
        }
        
        pu_obj = hwloc_get_obj_inside_cpuset_by_type(dt_topology, obj->cpuset, HWLOC_OBJ_PU, 0);
        printf("let's skip the first core\n");
        pu_obj = pu_obj->parent->next_cousin->first_child;
        hwloc_bitmap_asprintf(&str, pu_obj->cpuset);
        printf("main thread cpuset %s, obj name %s\n", str, pu_obj->name);
        dt_ht_desc[0].cpuset = pu_obj->cpuset;
        dt_ht_desc[0].TID = pthread_self();
        hwloc_set_thread_cpubind(dt_topology, (hwloc_thread_t)dt_ht_desc[0].TID, dt_ht_desc[0].cpuset, HWLOC_CPUBIND_THREAD);
        
        pthread_cond_init(&(dt_ht_pool.q_notempty), NULL);
        pthread_mutex_init(&(dt_ht_pool.ht_lock), NULL);
        pthread_barrier_init(&(dt_ht_pool.ht_barrier), NULL, dt_ht_pool.num_ht);
        for( _i = 0; _i < dt_ht_pool.num_ht; _i++ ) {
            assert(pu_obj != NULL);
            dt_ht_desc[_i].ht_pool = &dt_ht_pool;
            dt_ht_desc[_i].thread_id = _i;
            dt_ht_desc[_i].ht_data = NULL;
            dt_ht_desc[_i].task = NULL;
            dt_ht_desc[_i].status = 0;
            dt_ht_desc[_i].cpuset = pu_obj->cpuset;
            if (_i != 0) {
                rc = pthread_create(&(dt_ht_desc[_i].TID), NULL, opal_datatype_ht, (void *)&dt_ht_desc[_i]);
                if (rc) {
                    printf("ERROR; return code from pthread_create() is %d\n", rc);
                    datatype_ht_init = 2;
                    return rc;
                }
            }
            dt_ht_desc[_i].status = 1;
#if defined(DATATYPE_USE_HT)
            pu_obj = pu_obj->next_sibling;
#else
            pu_obj = pu_obj->parent->next_cousin->first_child;
#endif
        }
        datatype_ht_init = 1;
        dt_ht_pool.shutdown = 0;
    }
//    sleep(5);
    return OPAL_SUCCESS;
}

int32_t opal_datatype_ht_fini(void)
{
    uint32_t _i;
    if (datatype_ht_init != 0) {
        dt_ht_pool.shutdown = 1;
        pthread_cond_broadcast(&(dt_ht_pool.q_notempty));
    }
    for( _i = 1; _i < dt_ht_pool.num_ht; _i++ ) {
        if (dt_ht_desc[_i].status == 1) {
            pthread_join(dt_ht_desc[_i].TID, NULL);
        }
    }
    if (datatype_ht_init != 0) { 
        pthread_cond_destroy(&(dt_ht_pool.q_notempty));
        pthread_mutex_destroy(&(dt_ht_pool.ht_lock));
        pthread_barrier_destroy(&(dt_ht_pool.ht_barrier));
        datatype_ht_init = 0;
    }
    return OPAL_SUCCESS;
}

void opal_datatype_ht_pack_contiguous_loop_function(datatype_ht_desc_t* ht_desc)
{
    volatile datatype_ht_pool_t* ht_pool;
    volatile contiguous_loop_ht_data_t* ht_data;
    unsigned char* _destination;
    unsigned char* _source;
    uint32_t _copy_loops;
    ddt_endloop_desc_t* _end_loop;
    ddt_loop_desc_t*_loop;
    opal_convertor_t* CONVERTOR;
    size_t* SPACE; 
    int32_t _i, _j, begi, endi, chunk_size;
    
    ht_pool = ht_desc->ht_pool;
    ht_desc->num_task_done = 0;
    opal_atomic_add_32(&(ht_pool->ht_wake), 1);
    ht_data = (contiguous_loop_ht_data_t*)ht_desc->ht_data;
    SPACE = ht_data->SPACE;
    CONVERTOR = ht_data->CONVERTOR;
    _end_loop = ht_data->end_loop;
    _loop = ht_data->loop;
    _copy_loops = ht_data->copy_loops;
    _source = ht_data->source;
    _destination = ht_data->destination;
    
    chunk_size = _copy_loops / DATATYPE_CUT; 
    
    while (ht_pool->loop_unfini > 0) { 
        endi = opal_atomic_sub_32(&(ht_pool->loop_unfini), chunk_size);
        begi = endi + chunk_size - 1;
//        printf("unfini %d， chunk_size %d, beg %d, end %d\n", ht_pool->loop_unfini, chunk_size, begi, endi);
        for (_i = begi; _i >= endi; _i--) {
            if (_i < 0) {
                break;
            }
        
          // printf("tid %d, src %p [base %p], dst %p [base %p], cl %d extent %ld size %lu loop_ct %d\n",
          //          ht_desc->thread_id, _source, ht_data->source, _destination,
          //          ht_data->destination, _copy_loops, _loop->extent, _end_loop->size, ht_pool->loop_unfini);
               

            _j = _copy_loops - (_i + 1);
            _source = ht_data->source + _j * _loop->extent;
            _destination = ht_data->destination + _j * _end_loop->size;
            OPAL_DATATYPE_SAFEGUARD_POINTER( _source, _loop->extent, (CONVERTOR)->pBaseBuf,
                                            (CONVERTOR)->pDesc, (CONVERTOR)->count );
            //OPAL_DATATYPE_SAFEGUARD_POINTER( _destination, _end_loop->size, ht_data->destination,
            //                                (CONVERTOR)->pDesc, (CONVERTOR)->count );
            DO_DEBUG( opal_output( 0, "pack 3. memcpy( %p, %p, %lu ) => space %lu\n",
                                   _destination, _source, (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _j * _end_loop->size) ); );
            // printf ("tid %d, pack 3. memcpy( %p, %p, %lu ) => space %lu, i %d\n",
            //     ht_desc->thread_id, _destination, _source, (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _j * _end_loop->size), _j );
            MEMCPY_CSUM( _destination, _source, _end_loop->size, (CONVERTOR) );   
            ht_desc->num_task_done ++;    
        }
        chunk_size = ht_pool->loop_unfini / DATATYPE_CUT;
        if (chunk_size < DATATYPE_CHUNK_SIZE_LIMIT) {
            chunk_size = DATATYPE_CHUNK_SIZE_LIMIT;
        }
    }
    printf("pthread %d, task %d\n", ht_desc->thread_id, ht_desc->num_task_done);
    opal_atomic_add_32(&(ht_pool->ht_fini), 1);
    
}

void* opal_datatype_ht(void* hd)
{
    datatype_ht_desc_t* ht_desc;
    datatype_ht_pool_t* ht_pool;
    
    ht_desc = (datatype_ht_desc_t *)hd;
    ht_pool = ht_desc->ht_pool;
    opal_datatype_bind_thread(ht_desc);
    printf("Begin thread #%d, total %d!\n", ht_desc->thread_id, ht_pool->num_ht);
    while (1) {
        pthread_mutex_lock(&(ht_pool->ht_lock));
        while (ht_pool->loop_unfini <= 0 && ht_pool->shutdown == 0) {
            pthread_cond_wait(&(ht_pool->q_notempty), &(ht_pool->ht_lock)); 
    //      printf("thread %d wake up, ct %d\n", ht_desc->thread_id, ht_pool->loop_unfini);
        }
        pthread_mutex_unlock(&(ht_pool->ht_lock));
        if (ht_pool->shutdown) {
            break;
        }
        if (ht_desc->task != NULL) {
            ht_desc->task(ht_desc);
        }
    }
    return NULL;
}