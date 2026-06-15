//! Canonical DbgKernelId values — keep in sync with shaders/include/dgq_kernels.metal.

#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DbgKernelId {
    Unknown = 0,
    EmbedGather = 1,
    HalfToF32 = 2,
    F32ToHalf = 3,
    MemzeroBytes = 4,
    RmsNormRows = 5,
    RmsNormRowsTiled = 6,
    ResidualHalf = 7,
    ResidualF32b = 8,
    Gelu = 9,
    Swiglu = 10,
    SwigluMoeGateUp = 11,
    HalfScale = 12,
    SoftcapHalf = 13,
    GatherRows = 14,
    GatherProbCols = 15,
    GemmBlock = 16,
    GemmBlockGrouped = 17,
    GemmLinearGrouped = 18,
    GemmLinearF32 = 19,
    GemmQ4 = 20,
    GemmQ8 = 21,
    GemmQ8Rowk = 22,
    GemmNvfp4 = 23,
    DequantBlockMatrix = 24,
    QkRopeKv = 25,
    Attention = 26,
    MoeRouter = 27,
    MoeBucketCount = 28,
    MoeBucketFill = 29,
    MoeGrouped = 30,
    MoeScatterWeighted = 31,
    RouterTopKRows = 32,
    RouterScaleRows = 33,
    ScProbs = 34,
    ScSoftembed = 35,
    SoftmaxRows = 36,
    SampleRowstats = 37,
    SampleCommit = 38,
    SampleApply = 39,
    SampleWrite = 40,
    LogitRowstats = 41,
    PackEncoderKv = 42,
    VecAddInplace = 43,
    VecScaleInplace = 44,
    VecMulInplace = 45,
    VecFillZero = 46,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DbgErrorCode {
    None = 0,
    IndexOob = 1,
    RouteTokenOob = 2,
    SoftmaxNormZero = 3,
    NonFinite = 4,
    EntropyTooHigh = 5,
    QuantUnsupported = 6,
    ArenaOob = 7,
    BadDim = 8,
}
