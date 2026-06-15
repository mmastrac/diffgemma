#!/usr/bin/env swift
// MetalPerformanceShaders matmul oracle for bench-gemm (Q2.5).
// Usage: swift bench/mpsgraph_gemm.swift M K N ITERS
// C[M,N] = A[M,K] @ W[N,K]^T  (PyTorch linear layout), f32 activations/weights.
// Prints one line: GFLOP/s (resident buffers, batched encodes, single sync).

import Foundation
import Metal
import MetalPerformanceShaders

guard CommandLine.arguments.count == 5,
      let m = Int(CommandLine.arguments[1]),
      let k = Int(CommandLine.arguments[2]),
      let n = Int(CommandLine.arguments[3]),
      let iters = Int(CommandLine.arguments[4]) else {
    fputs("usage: mpsgraph_gemm.swift M K N ITERS\n", stderr)
    exit(1)
}

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!

let aRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: k, dataType: .float32)
let wRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: k, dataType: .float32)
let cRowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: n, dataType: .float32)

let aBuf = device.makeBuffer(length: m * aRowBytes, options: .storageModeShared)!
let wBuf = device.makeBuffer(length: n * wRowBytes, options: .storageModeShared)!
let cBuf = device.makeBuffer(length: m * cRowBytes, options: .storageModeShared)!

let aDesc = MPSMatrixDescriptor(rows: m, columns: k, rowBytes: aRowBytes, dataType: .float32)
let wDesc = MPSMatrixDescriptor(rows: n, columns: k, rowBytes: wRowBytes, dataType: .float32)
let cDesc = MPSMatrixDescriptor(rows: m, columns: n, rowBytes: cRowBytes, dataType: .float32)

let aMat = MPSMatrix(buffer: aBuf, descriptor: aDesc)
let wMat = MPSMatrix(buffer: wBuf, descriptor: wDesc)
let cMat = MPSMatrix(buffer: cBuf, descriptor: cDesc)

// W stored [N,K]; transposeRight yields [K,N] for multiply.
let kernel = MPSMatrixMultiplication(
    device: device,
    transposeLeft: false,
    transposeRight: true,
    resultRows: m,
    resultColumns: n,
    interiorColumns: k,
    alpha: 1.0,
    beta: 0.0
)

func run(count: Int) {
    guard let cmd = queue.makeCommandBuffer() else { return }
    for _ in 0..<count {
        kernel.encode(commandBuffer: cmd, leftMatrix: aMat, rightMatrix: wMat, resultMatrix: cMat)
    }
    cmd.commit()
    cmd.waitUntilCompleted()
}

let warmup = 3
run(count: warmup)

let t0 = CFAbsoluteTimeGetCurrent()
run(count: iters)
let elapsed = CFAbsoluteTimeGetCurrent() - t0

let flops = 2.0 * Double(m * k * n * iters)
let gflops = flops / elapsed / 1e9
print(String(format: "%.1f", gflops))
