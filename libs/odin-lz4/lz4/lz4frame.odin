package lz4

import "core:c"

when ODIN_OS == .Linux {
    when ODIN_ARCH == .amd64 {
        foreign import lz4lib "lz4_linux_x64.a"
    }
}
when ODIN_OS == .Darwin {
    when ODIN_ARCH == .amd64 || ODIN_ARCH == .arm64 {
        foreign import lz4lib "lz4_macos_universal.a"
    }
}
when ODIN_OS == .Windows {
    when ODIN_ARCH == .amd64 {
        when ODIN_DEBUG {
            foreign import lz4lib "lz4_windows_x64_debug.lib"
        } else {
            foreign import lz4lib "lz4_windows_x64_release.lib"
        }
    }
}

// LZ4F_VERSION  This number can be used to check for an incompatible API breaking change
LZ4F_VERSION :: 100

// Frame header constants
LZ4F_HEADER_SIZE_MIN  :: 7
LZ4F_HEADER_SIZE_MAX :: 19
LZ4F_BLOCK_HEADER_SIZE :: 4
LZ4F_BLOCK_CHECKSUM_SIZE :: 4
LZ4F_CONTENT_CHECKSUM_SIZE :: 4
LZ4F_MAGICNUMBER :: 0x184D2204
LZ4F_MAGIC_SKIPPABLE_START :: 0x184D2A50
LZ4F_MIN_SIZE_TO_KNOW_HEADER_LENGTH :: 5

// Block size IDs
LZ4F_BlockSizeID :: enum c.int {
    default_ = 0,
    max64KB  = 4,
    max256KB = 5,
    max1MB   = 6,
    max4MB   = 7,
}

// Block modes
LZ4F_BlockMode :: enum c.int {
    blockLinked    = 0,
    blockIndependent,
}

// Content checksum
LZ4F_ContentChecksum :: enum c.int {
    noContentChecksum = 0,
    contentChecksumEnabled,
}

// Block checksum
LZ4F_BlockChecksum :: enum c.int {
    noBlockChecksum = 0,
    blockChecksumEnabled,
}

// Frame type
LZ4F_FrameType :: enum c.int {
    frame          = 0,
    skippableFrame,
}

// Frame info
LZ4F_FrameInfo :: struct {
    blockSizeID:         LZ4F_BlockSizeID,
    blockMode:           LZ4F_BlockMode,
    contentChecksumFlag: LZ4F_ContentChecksum,
    frameType:           LZ4F_FrameType,
    contentSize:         u64,
    dictID:              u32,
    blockChecksumFlag:   LZ4F_BlockChecksum,
}

// Preferences
LZ4F_Preferences :: struct {
    frameInfo:        LZ4F_FrameInfo,
    compressionLevel: c.int,
    autoFlush:        c.uint,
    favorDecSpeed:    c.uint,
    reserved:         [3]c.uint,
}

// Compress options
LZ4F_CompressOptions :: struct {
    stableSrc: c.uint,
    reserved:  [3]c.uint,
}

// Decompress options
LZ4F_DecompressOptions :: struct {
    stableDst:     c.uint,
    skipChecksums: c.uint,
    reserved1:     c.uint,
    reserved0:     c.uint,
}

// Error codes
LZ4F_ErrorCodes :: enum c.int {
    OK_NoError = 0,
    ERROR_GENERIC,
    ERROR_maxBlockSize_invalid,
    ERROR_blockMode_invalid,
    ERROR_contentChecksumFlag_invalid,
    ERROR_compressionLevel_invalid,
    ERROR_headerVersion_wrong,
    ERROR_blockChecksum_invalid,
    ERROR_reservedFlag_set,
    ERROR_allocation_failed,
    ERROR_srcSize_tooLarge,
    ERROR_dstMaxSize_tooSmall,
    ERROR_frameHeader_incomplete,
    ERROR_frameType_unknown,
    ERROR_frameSize_wrong,
    ERROR_srcPtr_wrong,
    ERROR_decompressionFailed,
    ERROR_headerChecksum_invalid,
    ERROR_contentChecksum_invalid,
    ERROR_frameDecoding_alreadyStarted,
    ERROR_compressionState_uninitialized,
    ERROR_parameter_null,
    ERROR_maxCode,
}

// Opaque context types
LZ4F_CCtx :: distinct rawptr
LZ4F_DCtx :: distinct rawptr

@(default_calling_convention = "c")
foreign lz4lib {
    // Error management
    LZ4F_isError :: proc(code: c.size_t) -> c.uint ---
    LZ4F_getErrorName :: proc(code: c.size_t) -> cstring ---

    // Compression level max
    LZ4F_compressionLevel_max :: proc() -> c.int ---

    // One-shot compression
    LZ4F_compressFrameBound :: proc(srcSize: c.size_t, preferencesPtr: ^LZ4F_Preferences) -> c.size_t ---
    LZ4F_compressFrame :: proc(
        dstBuffer: rawptr, dstCapacity: c.size_t,
        srcBuffer: rawptr, srcSize: c.size_t,
        preferencesPtr: ^LZ4F_Preferences,
    ) -> c.size_t ---

    // Compression context management
    LZ4F_createCompressionContext :: proc(cctxPtr: ^LZ4F_CCtx, version: c.uint) -> c.size_t ---
    LZ4F_freeCompressionContext :: proc(cctx: LZ4F_CCtx) -> c.size_t ---

    // Streaming compression
    LZ4F_compressBegin :: proc(
        cctx: LZ4F_CCtx,
        dstBuffer: rawptr, dstCapacity: c.size_t,
        preferencesPtr: ^LZ4F_Preferences,
    ) -> c.size_t ---

    LZ4F_compressUpdate :: proc(
        cctx: LZ4F_CCtx,
        dstBuffer: rawptr, dstCapacity: c.size_t,
        srcBuffer: rawptr, srcSize: c.size_t,
        cOptPtr: ^LZ4F_CompressOptions,
    ) -> c.size_t ---

    LZ4F_flush :: proc(
        cctx: LZ4F_CCtx,
        dstBuffer: rawptr, dstCapacity: c.size_t,
        cOptPtr: ^LZ4F_CompressOptions,
    ) -> c.size_t ---

    LZ4F_compressEnd :: proc(
        cctx: LZ4F_CCtx,
        dstBuffer: rawptr, dstCapacity: c.size_t,
        cOptPtr: ^LZ4F_CompressOptions,
    ) -> c.size_t ---

    // Decompression context management
    LZ4F_createDecompressionContext :: proc(dctxPtr: ^LZ4F_DCtx, version: c.uint) -> c.size_t ---
    LZ4F_freeDecompressionContext :: proc(dctx: LZ4F_DCtx) -> c.size_t ---

    // Frame info extraction
    LZ4F_headerSize :: proc(src: rawptr, srcSize: c.size_t) -> c.size_t ---
    LZ4F_getFrameInfo :: proc(
        dctx: LZ4F_DCtx,
        frameInfoPtr: ^LZ4F_FrameInfo,
        srcBuffer: rawptr,
        srcSizePtr: ^c.size_t,
    ) -> c.size_t ---

    // Streaming decompression
    LZ4F_decompress :: proc(
        dctx: LZ4F_DCtx,
        dstBuffer: rawptr, dstSizePtr: ^c.size_t,
        srcBuffer: rawptr, srcSizePtr: ^c.size_t,
        dOptPtr: ^LZ4F_DecompressOptions,
    ) -> c.size_t ---

    LZ4F_resetDecompressionContext :: proc(dctx: LZ4F_DCtx) ---
    LZ4F_getVersion :: proc() -> c.uint ---
}
