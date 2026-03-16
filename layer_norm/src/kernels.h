
void layer_norm_simple_async(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2);

void layer_norm_simple_dsmem(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2);

void layer_norm_simple(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2);

void layer_norm_simple_dsmem_async_v1(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2);

void layer_norm_simple_dsmem_async_v2(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2);

void layer_norm_wellford(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2);

void layer_norm_simple_persistant(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2);

