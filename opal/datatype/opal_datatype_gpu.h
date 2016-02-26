#ifndef OPAL_DATATYPE_GPU_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_GPU_H_HAS_BEEN_INCLUDED

int32_t opal_datatype_gpu_init(void);
int32_t opal_datatype_gpu_fini(void);

extern void (*opal_datatype_cuda_init_p)(void);

extern void (*opal_datatype_cuda_fini_p)(void);

extern int32_t (*opal_generic_simple_pack_function_cuda_p)( opal_convertor_t* pConvertor,
                                                            struct iovec* iov, 
                                                            uint32_t* out_size,
                                                            size_t* max_data );
                                                            
extern int32_t (*opal_generic_simple_unpack_function_cuda_p)( opal_convertor_t* pConvertor,
                                                              struct iovec* iov, 
                                                              uint32_t* out_size,
                                                              size_t* max_data );
                                                              
extern void (*pack_contiguous_loop_cuda_p)( dt_elem_desc_t* ELEM,
                                            uint32_t* COUNT,
                                            unsigned char** SOURCE,
                                            unsigned char** DESTINATION,
                                            size_t* SPACE );
                                            
extern void (*unpack_contiguous_loop_cuda_p)( dt_elem_desc_t* ELEM,
                                             uint32_t* COUNT,
                                             unsigned char** SOURCE,
                                             unsigned char** DESTINATION,
                                             size_t* SPACE );

extern void (*pack_predefined_data_cuda_p)( dt_elem_desc_t* ELEM,
                                            uint32_t* COUNT,
                                            unsigned char** SOURCE,
                                            unsigned char** DESTINATION,
                                            size_t* SPACE );
                                            
extern void (*opal_cuda_sync_device_p)(void);
#endif /* OPAL_DATATYPE_GPU_H_HAS_BEEN_INCLUDED */