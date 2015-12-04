#define DATATYPE_CUT    3
#define DATATYPE_CHUNK_SIZE_LIMIT  10

#define OPAL_DATATYPE_PARALLEL 1

#ifndef OPAL_DATATYPE_HT_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_HT_H_HAS_BEEN_INCLUDED

#include <pthread.h>
#include "opal/mca/hwloc/hwloc.h"

#define NUM_HT  2
#define DATATYPE_USE_HT

typedef struct {
} datatype_ht_data_t;

typedef struct {
    datatype_ht_data_t super;
    unsigned char* destination;
    unsigned char* source;
    uint32_t copy_loops;
    ddt_loop_desc_t* loop;
    ddt_endloop_desc_t* end_loop;
    opal_convertor_t* CONVERTOR;
    size_t* SPACE; 
    unsigned char* callback_destination;
    unsigned char* callback_source;
} contiguous_loop_ht_data_t;

typedef struct {
    pthread_mutex_t ht_lock;
    pthread_barrier_t ht_barrier;
    pthread_cond_t q_notempty;
    uint32_t num_ht;
    int32_t loop_unfini;
    int32_t ht_fini;
    int32_t ht_wake;
    uint8_t shutdown;
} datatype_ht_pool_t;

struct datatype_ht_desc {
    pthread_t TID;
    hwloc_cpuset_t cpuset; 
    datatype_ht_pool_t* ht_pool;
    uint32_t thread_id;
    uint8_t status;
    datatype_ht_data_t* ht_data;
    void (*task)(struct datatype_ht_desc* ht_desc);
    uint32_t num_task_done;
};

typedef struct datatype_ht_desc datatype_ht_desc_t;

static uint8_t datatype_ht_init = 0;
extern datatype_ht_desc_t dt_ht_desc[NUM_HT];
extern datatype_ht_pool_t dt_ht_pool;
extern contiguous_loop_ht_data_t cl_ht_data;

int32_t opal_datatype_ht_init(void);
int32_t opal_datatype_ht_fini(void);
void* opal_datatype_ht(void* tid);
void* opal_datatype_ht_checksum(void* tid);
void opal_datatype_ht_pack_contiguous_loop(datatype_ht_desc_t* ht_desc);
void opal_datatype_ht_pack_contiguous_loop_checksum(datatype_ht_desc_t* ht_desc);

#endif  /* OPAL_DATATYPE_HT_H_HAS_BEEN_INCLUDED */