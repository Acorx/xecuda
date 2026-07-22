/**
 * @file xecuda_xmx.h
 * @brief Hardware Xe Matrix Extensions (XMX) Joint Matrix Operations for Intel Arc Xe2
 */

#ifndef XECUDA_XMX_H
#define XECUDA_XMX_H

#include "xecuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// XMX Systolic Tile Dimension Enum
typedef enum xeXmxTileSize {
    XE_XMX_TILE_8x8 = 0,
    XE_XMX_TILE_16x16 = 1,
    XE_XMX_TILE_32x32 = 2
} xeXmxTileSize_t;

// XMX Joint Matrix Handle
typedef struct xeXmxMatrixTile {
    int rows;
    int cols;
    xeXmxTileSize_t tileSize;
    void* dataPtr;
} xeXmxMatrixTile_t;

/**
 * @brief Initialize XMX Joint Matrix Tile for FP16/BF16/INT8 hardware execution
 */
xeCudaError_t xeXmxInitTile(xeXmxMatrixTile_t* tile, int rows, int cols, xeXmxTileSize_t tileSize);

/**
 * @brief Execute Systolic Joint Matrix Multiply Accumulate (MMA) on Intel Xe2 XMX
 * Computes: C = A * B + C
 */
xeCudaError_t xeXmxMultiplyAccumulate(
    const xeXmxMatrixTile_t* tileA,
    const xeXmxMatrixTile_t* tileB,
    xeXmxMatrixTile_t* tileC
);

#ifdef __cplusplus
}
#endif

#endif // XECUDA_XMX_H
