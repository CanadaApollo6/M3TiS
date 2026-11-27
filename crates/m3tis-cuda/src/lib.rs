//! Thin, zero-overhead CUDA Driver API (`libcuda.so.1`) wrapper for M3TiS.
//!
//! Uses direct `dlopen`/`dlsym` (zero cudart/bindgen bloat), RAII ownership for `DeviceBuf`,
//! `PinnedHostBuf` (`cuMemHostAlloc` 4K-aligned for zero-copy `O_DIRECT` + async PCIe DMA),
//! `Stream`, `Event`, and embedded `sm_86` `.cubin` kernels compiled by `nvcc`.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::ptr;
use std::sync::Arc;

pub mod dflash2;
pub mod exl3;
pub mod exl3_requant;
pub mod glm_ops;
pub mod linear;

pub type CuResult = c_int;
pub type CuDevice = c_int;
pub type CuContext = *mut c_void;
pub type CuStream = *mut c_void;
pub type CuEvent = *mut c_void;
pub type CuModule = *mut c_void;
pub type CuFunction = *mut c_void;
pub type CuDevicePtr = u64;

const CUDA_SUCCESS: CuResult = 0;
const CU_STREAM_NON_BLOCKING: c_uint = 1;
const CU_EVENT_DISABLE_TIMING: c_uint = 2;
const CU_MEMHOSTALLOC_PORTABLE: c_uint = 1;
const CU_MEMHOSTALLOC_DEVICEMAP: c_uint = 2;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

const RTLD_LAZY: c_int = 1;
const RTLD_LOCAL: c_int = 0;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[derive(Debug, Clone)]
pub struct CudaError {
    pub op: &'static str,
    pub code: CuResult,
    pub msg: String,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA {} failed (code {}): {}", self.op, self.code, self.msg)
    }
}
impl std::error::Error for CudaError {}

pub type Result<T> = std::result::Result<T, CudaError>;

#[allow(non_snake_case)]
struct DriverApi {
    cuInit: unsafe extern "C" fn(c_uint) -> CuResult,
    cuDeviceGet: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
    cuDevicePrimaryCtxRelease_v2: unsafe extern "C" fn(CuDevice) -> CuResult,
    cuCtxSetCurrent: unsafe extern "C" fn(CuContext) -> CuResult,
    cuMemGetInfo_v2: unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult,
    cuMemAlloc_v2: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    cuMemFree_v2: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    cuMemHostAlloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CuResult,
    cuMemFreeHost: unsafe extern "C" fn(*mut c_void) -> CuResult,
    cuMemsetD32Async: unsafe extern "C" fn(CuDevicePtr, c_uint, usize, CuStream) -> CuResult,
    cuMemcpyHtoDAsync_v2:
        unsafe extern "C" fn(CuDevicePtr, *const c_void, usize, CuStream) -> CuResult,
    cuMemcpyDtoHAsync_v2:
        unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize, CuStream) -> CuResult,
    cuMemcpyDtoDAsync_v2:
        unsafe extern "C" fn(CuDevicePtr, CuDevicePtr, usize, CuStream) -> CuResult,
    cuStreamCreate: unsafe extern "C" fn(*mut CuStream, c_uint) -> CuResult,
    cuStreamDestroy_v2: unsafe extern "C" fn(CuStream) -> CuResult,
    cuStreamSynchronize: unsafe extern "C" fn(CuStream) -> CuResult,
    cuStreamWaitEvent: unsafe extern "C" fn(CuStream, CuEvent, c_uint) -> CuResult,
    cuEventCreate: unsafe extern "C" fn(*mut CuEvent, c_uint) -> CuResult,
    cuEventDestroy_v2: unsafe extern "C" fn(CuEvent) -> CuResult,
    cuEventRecord: unsafe extern "C" fn(CuEvent, CuStream) -> CuResult,
    cuEventSynchronize: unsafe extern "C" fn(CuEvent) -> CuResult,
    cuModuleLoadData: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    cuModuleUnload: unsafe extern "C" fn(CuModule) -> CuResult,
    cuModuleGetFunction:
        unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    cuLaunchKernel: unsafe extern "C" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CuStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    cuGetErrorString: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
}

unsafe fn load_sym<T>(lib: *mut c_void, name: &str) -> Result<T> {
    let cname = CString::new(name).unwrap();
    let ptr = unsafe { dlsym(lib, cname.as_ptr()) };
    if ptr.is_null() {
        return Err(CudaError {
            op: "dlsym",
            code: -1,
            msg: format!("missing symbol {name} in libcuda.so.1"),
        });
    }
    Ok(unsafe { std::mem::transmute_copy(&ptr) })
}

impl DriverApi {
    fn load() -> Result<Self> {
        let libname = CString::new("libcuda.so.1").unwrap();
        let lib = unsafe { dlopen(libname.as_ptr(), RTLD_LAZY | RTLD_LOCAL) };
        if lib.is_null() {
            return Err(CudaError {
                op: "dlopen",
                code: -1,
                msg: "failed to load libcuda.so.1".into(),
            });
        }
        unsafe {
            Ok(Self {
                cuInit: load_sym(lib, "cuInit")?,
                cuDeviceGet: load_sym(lib, "cuDeviceGet")?,
                cuDeviceGetName: load_sym(lib, "cuDeviceGetName")?,
                cuDeviceGetAttribute: load_sym(lib, "cuDeviceGetAttribute")?,
                cuDevicePrimaryCtxRetain: load_sym(lib, "cuDevicePrimaryCtxRetain")?,
                cuDevicePrimaryCtxRelease_v2: load_sym(lib, "cuDevicePrimaryCtxRelease_v2")?,
                cuCtxSetCurrent: load_sym(lib, "cuCtxSetCurrent")?,
                cuMemGetInfo_v2: load_sym(lib, "cuMemGetInfo_v2")?,
                cuMemAlloc_v2: load_sym(lib, "cuMemAlloc_v2")?,
                cuMemFree_v2: load_sym(lib, "cuMemFree_v2")?,
                cuMemHostAlloc: load_sym(lib, "cuMemHostAlloc")?,
                cuMemFreeHost: load_sym(lib, "cuMemFreeHost")?,
                cuMemsetD32Async: load_sym(lib, "cuMemsetD32Async")?,
                cuMemcpyHtoDAsync_v2: load_sym(lib, "cuMemcpyHtoDAsync_v2")?,
                cuMemcpyDtoHAsync_v2: load_sym(lib, "cuMemcpyDtoHAsync_v2")?,
                cuMemcpyDtoDAsync_v2: load_sym(lib, "cuMemcpyDtoDAsync_v2")?,
                cuStreamCreate: load_sym(lib, "cuStreamCreate")?,
                cuStreamDestroy_v2: load_sym(lib, "cuStreamDestroy_v2")?,
                cuStreamSynchronize: load_sym(lib, "cuStreamSynchronize")?,
                cuStreamWaitEvent: load_sym(lib, "cuStreamWaitEvent")?,
                cuEventCreate: load_sym(lib, "cuEventCreate")?,
                cuEventDestroy_v2: load_sym(lib, "cuEventDestroy_v2")?,
                cuEventRecord: load_sym(lib, "cuEventRecord")?,
                cuEventSynchronize: load_sym(lib, "cuEventSynchronize")?,
                cuModuleLoadData: load_sym(lib, "cuModuleLoadData")?,
                cuModuleUnload: load_sym(lib, "cuModuleUnload")?,
                cuModuleGetFunction: load_sym(lib, "cuModuleGetFunction")?,
                cuLaunchKernel: load_sym(lib, "cuLaunchKernel")?,
                cuGetErrorString: load_sym(lib, "cuGetErrorString")?,
            })
        }
    }

    #[inline]
    fn check(&self, code: CuResult, op: &'static str) -> Result<()> {
        if code == CUDA_SUCCESS {
            return Ok(());
        }
        let mut s: *const c_char = ptr::null();
        let msg = unsafe {
            if (self.cuGetErrorString)(code, &mut s) == CUDA_SUCCESS && !s.is_null() {
                CStr::from_ptr(s).to_string_lossy().into_owned()
            } else {
                "unknown CUDA error".into()
            }
        };
        Err(CudaError { op, code, msg })
    }
}

/// Owned handle to a retained primary CUDA context on one GPU.
pub struct CudaContext {
    api: Arc<DriverApi>,
    device: CuDevice,
    ctx: CuContext,
    pub name: String,
    pub sm_major: i32,
    pub sm_minor: i32,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(device_ordinal: i32) -> Result<Arc<Self>> {
        let api = Arc::new(DriverApi::load()?);
        unsafe {
            api.check((api.cuInit)(0), "cuInit")?;
            let mut device: CuDevice = 0;
            api.check((api.cuDeviceGet)(&mut device, device_ordinal), "cuDeviceGet")?;
            let mut name_buf = [0u8; 256];
            api.check(
                (api.cuDeviceGetName)(name_buf.as_mut_ptr() as *mut c_char, 256, device),
                "cuDeviceGetName",
            )?;
            let name = CStr::from_ptr(name_buf.as_ptr() as *const c_char)
                .to_string_lossy()
                .into_owned();
            let mut sm_major = 0;
            let mut sm_minor = 0;
            api.check(
                (api.cuDeviceGetAttribute)(
                    &mut sm_major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                ),
                "cuDeviceGetAttribute(major)",
            )?;
            api.check(
                (api.cuDeviceGetAttribute)(
                    &mut sm_minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                ),
                "cuDeviceGetAttribute(minor)",
            )?;
            let mut ctx: CuContext = ptr::null_mut();
            api.check(
                (api.cuDevicePrimaryCtxRetain)(&mut ctx, device),
                "cuDevicePrimaryCtxRetain",
            )?;
            api.check((api.cuCtxSetCurrent)(ctx), "cuCtxSetCurrent")?;
            Ok(Arc::new(Self {
                api,
                device,
                ctx,
                name,
                sm_major,
                sm_minor,
            }))
        }
    }

    pub fn bind_current(&self) -> Result<()> {
        unsafe { self.api.check((self.api.cuCtxSetCurrent)(self.ctx), "cuCtxSetCurrent") }
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe {
            self.api
                .check((self.api.cuMemGetInfo_v2)(&mut free, &mut total), "cuMemGetInfo")?;
        }
        Ok((free, total))
    }

    pub fn stream(self: &Arc<Self>) -> Result<Stream> {
        let mut raw: CuStream = ptr::null_mut();
        unsafe {
            self.api
                .check((self.api.cuStreamCreate)(&mut raw, CU_STREAM_NON_BLOCKING), "cuStreamCreate")?;
        }
        Ok(Stream {
            ctx: Arc::clone(self),
            raw,
        })
    }

    pub fn event(self: &Arc<Self>) -> Result<Event> {
        let mut raw: CuEvent = ptr::null_mut();
        unsafe {
            self.api
                .check((self.api.cuEventCreate)(&mut raw, CU_EVENT_DISABLE_TIMING), "cuEventCreate")?;
        }
        Ok(Event {
            ctx: Arc::clone(self),
            raw,
        })
    }

    pub fn alloc_device(self: &Arc<Self>, bytes: usize) -> Result<DeviceBuf> {
        assert!(bytes > 0);
        let mut ptr: CuDevicePtr = 0;
        unsafe {
            self.api
                .check((self.api.cuMemAlloc_v2)(&mut ptr, bytes), "cuMemAlloc")?;
        }
        Ok(DeviceBuf {
            ctx: Arc::clone(self),
            ptr,
            len: bytes,
        })
    }

    /// Allocate 4K-aligned page-locked pinned host RAM (`cuMemHostAlloc`) suitable for both
    /// zero-copy Linux `O_DIRECT` `io_uring` DMA and async PCIe HtoD/DtoH transfers.
    pub fn alloc_pinned(self: &Arc<Self>, bytes: usize) -> Result<PinnedHostBuf> {
        assert!(bytes > 0);
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe {
            self.api.check(
                (self.api.cuMemHostAlloc)(
                    &mut ptr,
                    bytes,
                    CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP,
                ),
                "cuMemHostAlloc",
            )?;
        }
        Ok(PinnedHostBuf {
            ctx: Arc::clone(self),
            ptr: ptr as *mut u8,
            len: bytes,
        })
    }

    pub fn load_module(self: &Arc<Self>, image: &[u8]) -> Result<CudaModule> {
        let mut raw: CuModule = ptr::null_mut();
        unsafe {
            self.api.check(
                (self.api.cuModuleLoadData)(&mut raw, image.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
        }
        Ok(CudaModule {
            ctx: Arc::clone(self),
            raw,
        })
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.api.cuDevicePrimaryCtxRelease_v2)(self.device);
        }
    }
}

pub struct Stream {
    ctx: Arc<CudaContext>,
    pub raw: CuStream,
}

impl Stream {
    pub fn wait(&self, ev: &Event) -> Result<()> {
        unsafe {
            self.ctx
                .api
                .check((self.ctx.api.cuStreamWaitEvent)(self.raw, ev.raw, 0), "cuStreamWaitEvent")
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            self.ctx
                .api
                .check((self.ctx.api.cuStreamSynchronize)(self.raw), "cuStreamSynchronize")
        }
    }

    /// Async PCIe host-to-device copy from pinned memory offset to device pointer.
    ///
    /// # Safety
    /// Caller must ensure `src[..len]` remains valid and unmodified until `self` completes,
    /// and `dst_ptr..dst_ptr+len` is valid device memory.
    pub unsafe fn htod_async(&self, dst_ptr: CuDevicePtr, src: *const u8, len: usize) -> Result<()> {
        unsafe {
            self.ctx.api.check(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(dst_ptr, src as *const c_void, len, self.raw),
                "cuMemcpyHtoDAsync",
            )
        }
    }

    /// # Safety
    /// Caller must ensure `dst[..len]` remains valid until `self` completes.
    pub unsafe fn dtoh_async(&self, dst: *mut u8, src_ptr: CuDevicePtr, len: usize) -> Result<()> {
        unsafe {
            self.ctx.api.check(
                (self.ctx.api.cuMemcpyDtoHAsync_v2)(dst as *mut c_void, src_ptr, len, self.raw),
                "cuMemcpyDtoHAsync",
            )
        }
    }

    pub fn dtod_async(&self, dst_ptr: CuDevicePtr, src_ptr: CuDevicePtr, len: usize) -> Result<()> {
        unsafe {
            self.ctx.api.check(
                (self.ctx.api.cuMemcpyDtoDAsync_v2)(dst_ptr, src_ptr, len, self.raw),
                "cuMemcpyDtoDAsync",
            )
        }
    }

    pub fn memset_d32_async(&self, dst_ptr: CuDevicePtr, val: u32, count_u32: usize) -> Result<()> {
        unsafe {
            self.ctx.api.check(
                (self.ctx.api.cuMemsetD32Async)(dst_ptr, val, count_u32, self.raw),
                "cuMemsetD32Async",
            )
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.ctx.api.cuStreamDestroy_v2)(self.raw);
        }
    }
}

pub struct Event {
    ctx: Arc<CudaContext>,
    pub raw: CuEvent,
}

impl Event {
    pub fn record(&self, stream: &Stream) -> Result<()> {
        unsafe {
            self.ctx
                .api
                .check((self.ctx.api.cuEventRecord)(self.raw, stream.raw), "cuEventRecord")
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            self.ctx
                .api
                .check((self.ctx.api.cuEventSynchronize)(self.raw), "cuEventSynchronize")
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.ctx.api.cuEventDestroy_v2)(self.raw);
        }
    }
}

pub struct DeviceBuf {
    ctx: Arc<CudaContext>,
    ptr: CuDevicePtr,
    len: usize,
}

impl DeviceBuf {
    #[inline]
    pub fn ptr(&self) -> CuDevicePtr {
        self.ptr
    }

    #[inline]
    pub fn at(&self, byte_offset: usize) -> CuDevicePtr {
        debug_assert!(byte_offset <= self.len);
        self.ptr + byte_offset as u64
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.ctx.api.cuMemFree_v2)(self.ptr);
        }
    }
}

pub struct PinnedHostBuf {
    ctx: Arc<CudaContext>,
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for PinnedHostBuf {}

impl PinnedHostBuf {
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    #[inline]
    pub fn as_i32_mut(&mut self) -> &mut [i32] {
        assert_eq!(self.len % 4, 0);
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut i32, self.len / 4) }
    }

    #[inline]
    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        assert_eq!(self.len % 4, 0);
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut f32, self.len / 4) }
    }
}

impl Drop for PinnedHostBuf {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.ctx.api.cuMemFreeHost)(self.ptr as *mut c_void);
        }
    }
}

pub struct CudaModule {
    ctx: Arc<CudaContext>,
    raw: CuModule,
}

impl CudaModule {
    pub fn function(&self, name: &str) -> Result<CudaFunction> {
        let cname = CString::new(name).unwrap();
        let mut func: CuFunction = ptr::null_mut();
        unsafe {
            self.ctx.api.check(
                (self.ctx.api.cuModuleGetFunction)(&mut func, self.raw, cname.as_ptr()),
                "cuModuleGetFunction",
            )?;
        }
        Ok(CudaFunction {
            ctx: Arc::clone(&self.ctx),
            raw: func,
        })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.ctx.api.cuModuleUnload)(self.raw);
        }
    }
}

pub struct CudaFunction {
    ctx: Arc<CudaContext>,
    raw: CuFunction,
}

impl CudaFunction {
    /// # Safety
    /// Caller must match the kernel's exact C ABI parameter types and buffer lifetimes.
    pub unsafe fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: &Stream,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        unsafe {
            self.ctx.api.check(
                (self.ctx.api.cuLaunchKernel)(
                    self.raw,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared_mem,
                    stream.raw,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )
        }
    }
}

const ROUTER_REMAP_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/router_remap.cubin"));

/// Embedded `sm_86` router, DynTopK, FSR remap, resident-renorm, and DMA checksum kernels.
pub struct RouterRemapKernels {
    _module: CudaModule,
    remap: CudaFunction,
    remap_resident_renorm: CudaFunction,
    sigmoid_dyntopk: CudaFunction,
    xor_checksum: CudaFunction,
}

impl RouterRemapKernels {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = ctx.load_module(ROUTER_REMAP_CUBIN)?;
        let remap = module.function("m3tis_fsr_remap_i32")?;
        let remap_resident_renorm = module.function("m3tis_fsr_remap_resident_renorm")?;
        let sigmoid_dyntopk = module.function("m3tis_sigmoid_topk_dyntopk_f32")?;
        let xor_checksum = module.function("m3tis_slab_xor_checksum_u32")?;
        Ok(Self {
            _module: module,
            remap,
            remap_resident_renorm,
            sigmoid_dyntopk,
            xor_checksum,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn remap_resident_renorm(
        &self,
        stream: &Stream,
        mut in_picks: CuDevicePtr,
        mut in_wts: CuDevicePtr,
        mut out_picks: CuDevicePtr,
        mut out_wts: CuDevicePtr,
        mut layer_map: CuDevicePtr,
        mut rows: i32,
        mut top_k: i32,
        mut logical_experts: i32,
        mut skip_in: i32,
        mut skip_out: i32,
        mut target_scale: f32,
    ) -> Result<()> {
        let mut args: [*mut c_void; 11] = [
            &mut in_picks as *mut _ as *mut c_void,
            &mut in_wts as *mut _ as *mut c_void,
            &mut out_picks as *mut _ as *mut c_void,
            &mut out_wts as *mut _ as *mut c_void,
            &mut layer_map as *mut _ as *mut c_void,
            &mut rows as *mut _ as *mut c_void,
            &mut top_k as *mut _ as *mut c_void,
            &mut logical_experts as *mut _ as *mut c_void,
            &mut skip_in as *mut _ as *mut c_void,
            &mut skip_out as *mut _ as *mut c_void,
            &mut target_scale as *mut _ as *mut c_void,
        ];
        unsafe {
            self.remap_resident_renorm
                .launch((rows as u32, 1, 1), (32, 1, 1), 0, stream, &mut args)
        }
    }

    pub fn remap_i32(
        &self,
        stream: &Stream,
        mut in_picks: CuDevicePtr,
        mut out_picks: CuDevicePtr,
        mut layer_map: CuDevicePtr,
        mut count: i32,
        mut logical_experts: i32,
        mut skip_in: i32,
        mut skip_out: i32,
    ) -> Result<()> {
        let threads = 256u32;
        let blocks = ((count as u32) + threads - 1) / threads;
        let mut args: [*mut c_void; 7] = [
            &mut in_picks as *mut _ as *mut c_void,
            &mut out_picks as *mut _ as *mut c_void,
            &mut layer_map as *mut _ as *mut c_void,
            &mut count as *mut _ as *mut c_void,
            &mut logical_experts as *mut _ as *mut c_void,
            &mut skip_in as *mut _ as *mut c_void,
            &mut skip_out as *mut _ as *mut c_void,
        ];
        unsafe { self.remap.launch((blocks, 1, 1), (threads, 1, 1), 0, stream, &mut args) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sigmoid_topk_dyntopk(
        &self,
        stream: &Stream,
        logits: CuDevicePtr,
        bias: CuDevicePtr,
        out_picks: CuDevicePtr,
        out_wts: CuDevicePtr,
        rows: i32,
        experts: i32,
        top_k: i32,
        scale: f32,
        dyn_mass: f32,
        dyn_min_k: i32,
        shared_id: i32,
    ) -> Result<()> {
        self.sigmoid_topk_residency(
            stream, logits, bias, out_picks, out_wts, rows, experts, top_k, scale, dyn_mass,
            dyn_min_k, shared_id, 0, 0.0, top_k,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sigmoid_topk_residency(
        &self,
        stream: &Stream,
        mut logits: CuDevicePtr,
        mut bias: CuDevicePtr,
        mut out_picks: CuDevicePtr,
        mut out_wts: CuDevicePtr,
        mut rows: i32,
        mut experts: i32,
        mut top_k: i32,
        mut scale: f32,
        mut dyn_mass: f32,
        mut dyn_min_k: i32,
        mut shared_id: i32,
        mut layer_map: CuDevicePtr,
        mut residency_bias: f32,
        mut strict_k: i32,
    ) -> Result<()> {
        let mut args: [*mut c_void; 14] = [
            &mut logits as *mut _ as *mut c_void,
            &mut bias as *mut _ as *mut c_void,
            &mut out_picks as *mut _ as *mut c_void,
            &mut out_wts as *mut _ as *mut c_void,
            &mut rows as *mut _ as *mut c_void,
            &mut experts as *mut _ as *mut c_void,
            &mut top_k as *mut _ as *mut c_void,
            &mut scale as *mut _ as *mut c_void,
            &mut dyn_mass as *mut _ as *mut c_void,
            &mut dyn_min_k as *mut _ as *mut c_void,
            &mut shared_id as *mut _ as *mut c_void,
            &mut layer_map as *mut _ as *mut c_void,
            &mut residency_bias as *mut _ as *mut c_void,
            &mut strict_k as *mut _ as *mut c_void,
        ];
        unsafe {
            self.sigmoid_dyntopk
                .launch((rows as u32, 1, 1), (128, 1, 1), 0, stream, &mut args)
        }
    }

    pub fn xor_checksum_u32(
        &self,
        stream: &Stream,
        mut data: CuDevicePtr,
        mut out_xor: CuDevicePtr,
        mut n_words: u64,
    ) -> Result<()> {
        stream.memset_d32_async(out_xor, 0, 1)?;
        let blocks = 256u32;
        let threads = 256u32;
        let mut args: [*mut c_void; 3] = [
            &mut data as *mut _ as *mut c_void,
            &mut out_xor as *mut _ as *mut c_void,
            &mut n_words as *mut _ as *mut c_void,
        ];
        unsafe {
            self.xor_checksum
                .launch((blocks, 1, 1), (threads, 1, 1), 0, stream, &mut args)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_pinned_remap_and_fused_dyntopk_sm86() {
        let ctx = CudaContext::new(0).expect("open RTX 3090");
        assert!(ctx.sm_major >= 8, "expected sm_8x, got {}.{}", ctx.sm_major, ctx.sm_minor);
        let stream = ctx.stream().unwrap();
        let kernels = RouterRemapKernels::load(&ctx).expect("load sm_86 cubin");

        // 1. Verify pinned host alignment (4K) & fused sigmoid + DynTopK kernel
        let e = 288usize;
        let mut h_logits = ctx.alloc_pinned(e * 4).unwrap();
        assert_eq!((h_logits.as_ptr() as usize) % 4096, 0);
        for v in h_logits.as_f32_mut().iter_mut() {
            *v = -6.0;
        }
        // Make 4 dominant experts (mass >> 80%) and 4 weak tail experts
        for &idx in &[10usize, 42, 100, 250] {
            h_logits.as_f32_mut()[idx] = 5.0;
        }
        for &idx in &[1usize, 2, 3, 4] {
            h_logits.as_f32_mut()[idx] = -2.0;
        }
        let d_logits = ctx.alloc_device(e * 4).unwrap();
        let d_picks = ctx.alloc_device(9 * 4).unwrap();
        let d_wts = ctx.alloc_device(9 * 4).unwrap();
        let mut h_picks = ctx.alloc_pinned(9 * 4).unwrap();
        let mut h_wts = ctx.alloc_pinned(9 * 4).unwrap();

        unsafe {
            stream.htod_async(d_logits.ptr(), h_logits.as_ptr(), e * 4).unwrap();
        }
        kernels
            .sigmoid_topk_dyntopk(
                &stream,
                d_logits.ptr(),
                0,
                d_picks.ptr(),
                d_wts.ptr(),
                1,
                e as i32,
                8,
                2.5,
                0.80,
                4,
                288,
            )
            .unwrap();
        unsafe {
            stream.dtoh_async(h_picks.as_mut_ptr(), d_picks.ptr(), 9 * 4).unwrap();
            stream.dtoh_async(h_wts.as_mut_ptr(), d_wts.ptr(), 9 * 4).unwrap();
        }
        stream.synchronize().unwrap();

        let p = &h_picks.as_i32_mut()[..9];
        let w = &h_wts.as_f32_mut()[..9];
        // Top 4 kept, bottom 4 pruned to shared_id (288) with weight 0, slot 8 = shared (288, 1.0)
        assert_eq!(p[4..9], [288, 288, 288, 288, 288]);
        assert_eq!(w[4..8], [0.0, 0.0, 0.0, 0.0]);
        assert!((w[0..4].iter().sum::<f32>() - 2.5).abs() < 1e-4);
        assert_eq!(w[8], 1.0);

        // 2. Verify FSR remap kernel
        let mut h_map = ctx.alloc_pinned(e * 4).unwrap();
        h_map.as_i32_mut().fill(-1);
        h_map.as_i32_mut()[10] = 7;
        h_map.as_i32_mut()[42] = 19;
        h_map.as_i32_mut()[100] = 3;
        h_map.as_i32_mut()[250] = 55;
        let d_map = ctx.alloc_device(e * 4).unwrap();
        let d_mapped = ctx.alloc_device(9 * 4).unwrap();
        unsafe {
            stream.htod_async(d_map.ptr(), h_map.as_ptr(), e * 4).unwrap();
        }
        kernels
            .remap_i32(&stream, d_picks.ptr(), d_mapped.ptr(), d_map.ptr(), 9, 288, 288, 566)
            .unwrap();
        unsafe {
            stream.dtoh_async(h_picks.as_mut_ptr(), d_mapped.ptr(), 9 * 4).unwrap();
        }
        stream.synchronize().unwrap();
        let mut slots: Vec<i32> = h_picks.as_i32_mut()[0..4].to_vec();
        slots.sort();
        assert_eq!(slots, vec![3, 7, 19, 55]);
        assert_eq!(h_picks.as_i32_mut()[4..9], [566, 566, 566, 566, 566]);

        // 3. Verify zero-sync resident-only mask & renorm when 2 of 4 picked experts are absent (-1)
        h_map.as_i32_mut()[100] = -1;
        h_map.as_i32_mut()[250] = -1;
        unsafe {
            stream.htod_async(d_map.ptr(), h_map.as_ptr(), e * 4).unwrap();
        }
        let d_wts_out = ctx.alloc_device(9 * 4).unwrap();
        kernels
            .remap_resident_renorm(
                &stream,
                d_picks.ptr(),
                d_wts.ptr(),
                d_mapped.ptr(),
                d_wts_out.ptr(),
                d_map.ptr(),
                1,
                8,
                288,
                288,
                566,
                2.5,
            )
            .unwrap();
        unsafe {
            stream.dtoh_async(h_picks.as_mut_ptr(), d_mapped.ptr(), 9 * 4).unwrap();
            stream.dtoh_async(h_wts.as_mut_ptr(), d_wts_out.ptr(), 9 * 4).unwrap();
        }
        stream.synchronize().unwrap();
        let kept_slots: Vec<i32> = h_picks.as_i32_mut()[..8]
            .iter()
            .copied()
            .filter(|&s| s != 566)
            .collect();
        assert_eq!(kept_slots.len(), 2);
        let sum_kept: f32 = h_wts.as_f32_mut()[..8].iter().sum();
        assert!((sum_kept - 2.5).abs() < 1e-4, "renorm sum failed: {sum_kept}");
        assert_eq!(h_wts.as_f32_mut()[8], 1.0);
    }
}
