//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it.
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host.
//! * **Root CBVs and root SRVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` in [`ROOT_CBV_SPACE`], `i` the title's root parameter
//!   index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`]); the
//!   address the title sets is the buffer's.
//! * **Vertex fetch** (added 2026-11-23). Grand Theft Auto VI's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//!
//! * **Texture atomics** (added 2026-11-23): a texture UAV the shader uses
//!   atomically is declared a 32-bit unsigned single-channel view; the
//!   host's view at that index must be an R32 integer view (64-bit atomics
//!   stop the translation). Global loads from a root SRV read it raw.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Export shaders** ([`HwStage::Es`], emitted as `vs_6_6`). The vertex
//!   program compiled to run with a geometry shader. Its merged stage data
//!   is not read: `exp paramN` is signature row N, and a system VGPR read
//!   before it is written stops the translation instead of becoming zero.
//!   `s_setpc_b64` through a register ends the shader (the handover to the
//!   geometry half). A container that also has a plain vertex program uses
//!   that one. Geometry, hull and local programs are not translated.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).

mod bitcode;
mod container;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
mod lower;
mod sampler;
mod shader;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod valu_tests;
mod wave;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use sampler::StaticSampler;
pub use shader::{ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE};

/// The loop limit without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or
/// `0x` hex, at least 1) sets it instead. A loop that hits the limit in
/// every lane of a full-screen pass still runs for seconds (2^20 times a
/// 4K screen is about 10^13 iterations); a much lower limit shows whether
/// such loops are what makes a frame slow. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
        } else {
            lower::LOOP_LIMIT
        }
    })
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => lower::LOOP_LIMIT,
    }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the shader reads: `t<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u8>,
}

impl HostShader {
    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table` (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// Translates the vertex, pixel, compute, mesh or export program of a
/// console shader container. An export program is used only when the
/// container has none of the others: a vertex program compiled both ways
/// keeps the plain one.
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let programs = crate::native_programs(dxil.data)?;
    let native = programs
        .iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .or_else(|| programs.iter().find(|p| p.stage == HwStage::Es))
        .cloned()
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: if programs
                .iter()
                .any(|p| matches!(p.stage, HwStage::Gs | HwStage::Hs | HwStage::Ls))
            {
                "a geometry, hull or local stage"
            } else {
                "a container without a vertex, pixel, compute, mesh or export program"
            },
        })?;
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let res = resource::track(&p, &native.layout, native.stage)?;
    let io = io::raise(&p, &native)?;
    let st = structure::structure(&p)?;
    emit(&c, &native, &p, &st, &io, &res)
}

/// Emits a program whose earlier stages already ran.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say.
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (Grand Theft Auto VI's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let l = lower::lower(&lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        mesh_stride,
    })?;
    Ok(HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.filter |= 0x80;
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
    })
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask is
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A pixel shader writing a constant colour.
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res);
    }

    /// An export shader is the vertex program compiled for a geometry
    /// shader. It becomes a vertex shader: parameter N is signature row N,
    /// a handover to the geometry half ends it, and a system VGPR is not
    /// read as zero. A plain vertex program wins when both are present.
    /// Geometry, hull and local programs stay untranslated.
    #[test]
    fn an_export_shader_translates_as_a_vertex_shader() {
        let moves = [
            0x7E00_0280u32, // v_mov_b32 v0, 0
            0x7E02_0280,    // v1
            0x7E04_0280,    // v2
            0x7E06_0280,    // v3
        ];
        let exports = [
            0xF800_021Fu32, // exp param1, v0, v1, v2, v3
            0x0302_0100,
            0xF800_08CF, // exp pos0, v0, v1, v2, v3 done
            0x0302_0100,
        ];
        let end = 0xBF81_0000u32; // s_endpgm
        let handover = 0xBE80_2000u32; // s_setpc_b64 s[0:1]
        let mut ended = moves.to_vec();
        ended.extend(exports);
        ended.push(end);
        let mut handed = moves.to_vec();
        handed.extend(exports);
        handed.push(handover);

        let host = translate_stage(Hw::Es, &ended, &[]);
        assert_eq!(host.stage, crate::HwStage::Es);
        let program = dxil_program(&host.dxil);
        assert_eq!(program.kind, crate::dxil::ShaderKind::Vertex);
        assert_eq!(program.shader_model, (6, 6));
        let names = symbol_names(program.bitcode);
        assert!(
            names.iter().any(|n| n == "dx.op.storeOutput.f32"),
            "exports did not become vertex outputs: {names:?}"
        );

        let handed = translate_stage(Hw::Es, &handed, &[]);
        assert_eq!(handed.stage, crate::HwStage::Es);
        assert!(
            symbol_names(dxil_program(&handed.dxil).bitcode)
                .iter()
                .any(|n| n == "dx.op.storeOutput.f32")
        );

        // v0 is never written, so the position export reads a system VGPR.
        let reads_system = [0xF800_08CFu32, 0x0302_0100, end];
        let err = translate_stage_err(Hw::Es, &reads_system, &[]);
        assert!(
            err.to_string().contains("a merged-stage system VGPR"),
            "{err}"
        );

        let err = translate_stage_err(Hw::Gs, &[end], &[]);
        assert!(
            err.to_string().contains("a geometry, hull or local stage"),
            "{err}"
        );
        let err = translate_stage_err(Hw::Hs, &[end], &[]);
        assert!(
            err.to_string().contains("a geometry, hull or local stage"),
            "{err}"
        );
        let err = translate_stage_err(Hw::Ls, &[end], &[]);
        assert!(
            err.to_string().contains("a geometry, hull or local stage"),
            "{err}"
        );

        // Both variants: the plain vertex program is the one a pipeline
        // without a geometry shader runs.
        let both = translate_stage(Hw::Es, &ended, &[(Hw::Vs, &[end])]);
        assert_eq!(both.stage, crate::HwStage::Vs);
    }

    /// Which hardware stage a synthetic container carries.
    #[derive(Clone, Copy)]
    enum Hw {
        Es,
        Gs,
        Hs,
        Ls,
        Vs,
    }

    impl Hw {
        fn slot(self) -> usize {
            match self {
                Hw::Ls => 0x08,
                Hw::Hs => 0x10,
                Hw::Es => 0x18,
                Hw::Gs => 0x20,
                Hw::Vs => 0x28,
            }
        }
    }

    fn translate_stage(stage: Hw, code: &[u32], also: &[(Hw, &[u32])]) -> super::HostShader {
        match translate_stage_result(stage, code, also) {
            Ok(s) => s,
            Err(e) => panic!("{e}"),
        }
    }

    fn translate_stage_err(stage: Hw, code: &[u32], also: &[(Hw, &[u32])]) -> crate::ShaderError {
        match translate_stage_result(stage, code, also) {
            Err(e) => e,
            Ok(_) => panic!("translation succeeded"),
        }
    }

    fn translate_stage_result(
        stage: Hw,
        code: &[u32],
        also: &[(Hw, &[u32])],
    ) -> Result<super::HostShader, crate::ShaderError> {
        let mut programs = vec![(stage, code)];
        programs.extend(also.iter().map(|(s, c)| (*s, *c)));
        let dxil = native_part(&programs);
        let osg1 = signature_position_and_row1();
        let bytes = crate::dxbc::build(&[
            (crate::dxbc::FourCC::OSG1, &osg1),
            (crate::dxbc::FourCC::DXIL, &dxil),
        ]);
        super::translate(&bytes)
    }

    /// `OSG1`: `SV_Position`, then a generic row 1 (what `exp param1` writes).
    fn signature_position_and_row1() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(2u32.to_le_bytes());
        out.extend(8u32.to_le_bytes());
        let pos = 8 + 64;
        let tex = pos + "SV_Position".len() as u32 + 1;
        out.extend(sig_row(pos, 0, 1, 0));
        out.extend(sig_row(tex, 0, 0, 1));
        out.extend(b"SV_Position\0");
        out.extend(b"TEXCOORD\0");
        out
    }

    /// One signature row the parser accepts: stream 0, f32, mask `xyzw`.
    fn sig_row(name_at: u32, index: u32, kind: u32, register: u32) -> [u8; 32] {
        let mut row = [0u8; 32];
        row[4..8].copy_from_slice(&name_at.to_le_bytes());
        row[8..12].copy_from_slice(&index.to_le_bytes());
        row[12..16].copy_from_slice(&kind.to_le_bytes());
        row[16..20].copy_from_slice(&3u32.to_le_bytes());
        row[20..24].copy_from_slice(&register.to_le_bytes());
        row[24] = 0xF;
        row
    }

    /// A `DXIL` part whose native blob holds `programs` (stage, machine code).
    fn native_part(programs: &[(Hw, &[u32])]) -> Vec<u8> {
        const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;
        const RECORD: u64 = 0x54;
        // Enough zero stage data for a vertex program's semantic tables.
        const STAGE: u64 = 0x200;
        let mut payload = 6u64.to_le_bytes().to_vec();
        payload.resize(0x68, 0);
        for at in [0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x48, 0x60] {
            payload[at..at + 8].copy_from_slice(&ABSENT.to_le_bytes());
        }
        for &(stage, words) in programs {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let record = payload.len() as u64;
            let code_at = record + RECORD + STAGE;
            payload[stage.slot()..stage.slot() + 8].copy_from_slice(&record.to_le_bytes());
            payload.resize(code_at as usize, 0);
            let at = record as usize;
            payload[at..at + 8].copy_from_slice(&code_at.to_le_bytes());
            payload[at + 8..at + 16].copy_from_slice(&(code.len() as u64).to_le_bytes());
            // 16 VGPRs, 8 SGPRs.
            payload[at + 0x50..at + 0x54].copy_from_slice(&(16u32 | 8 << 20).to_le_bytes());
            payload.extend_from_slice(&code);
        }
        let mut fssm = b"FSSM".to_vec();
        fssm.extend_from_slice(&0x101u32.to_le_bytes());
        fssm.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        fssm.extend_from_slice(&payload);
        let stream = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
        let mut xbch = b"XBCH".to_vec();
        xbch.extend_from_slice(&3u32.to_le_bytes());
        xbch.extend_from_slice(&(stream.len() as u64).to_le_bytes());
        xbch.extend_from_slice(&(fssm.len() as u64).to_le_bytes());
        xbch.extend_from_slice(&stream);

        let mut out = vec![0u8; 0x18];
        out.extend_from_slice(b"XDAT");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&0x24u32.to_le_bytes());
        let offset = (out.len() + 12) as u32;
        for v in [0x0du32, offset, xbch.len() as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&xbch);
        out
    }

    fn dxil_program(container: &[u8]) -> crate::dxil::DxilProgram<'_> {
        let c = crate::dxbc::Container::parse(container).unwrap();
        let part = c.part(crate::dxbc::FourCC::DXIL).unwrap().data;
        crate::dxil::DxilProgram::parse(part).unwrap()
    }

    /// Value-symbol names of an unabbreviated LLVM 3.7 module.
    fn symbol_names(bitcode: &[u8]) -> Vec<String> {
        let mut words = Vec::with_capacity(bitcode.len() / 4);
        for chunk in bitcode.chunks(4) {
            words.push(u32::from_le_bytes(chunk.try_into().unwrap()));
        }
        let mut bits = Bits {
            words: &words,
            pos: 32,
        };
        assert_eq!(bits.take(2), 1, "bitcode does not open with a block");
        assert_eq!(bits.vbr(8), 8, "bitcode block is not the module");
        let width = bits.vbr(4) as u32;
        bits.align();
        let len = bits.take(32) as usize;
        let end = bits.pos + len * 32;
        let mut names = Vec::new();
        walk(&mut bits, width, end, false, &mut names);
        names
    }

    struct Bits<'a> {
        words: &'a [u32],
        pos: usize,
    }

    impl Bits<'_> {
        fn take(&mut self, width: u32) -> u64 {
            let mut v = 0u64;
            for i in 0..width {
                let p = self.pos + i as usize;
                let bit = u64::from((self.words[p / 32] >> (p % 32)) & 1);
                v |= bit << i;
            }
            self.pos += width as usize;
            v
        }

        fn vbr(&mut self, width: u32) -> u64 {
            let hi = 1u64 << (width - 1);
            let mut v = 0u64;
            let mut shift = 0;
            loop {
                let c = self.take(width);
                v |= (c & (hi - 1)) << shift;
                if c & hi == 0 {
                    return v;
                }
                shift += width - 1;
            }
        }

        fn align(&mut self) {
            let rem = self.pos % 32;
            if rem != 0 {
                self.pos += 32 - rem;
            }
        }
    }

    fn walk(bits: &mut Bits<'_>, width: u32, end: usize, symbols: bool, names: &mut Vec<String>) {
        while bits.pos < end {
            match bits.take(width) {
                0 => {
                    bits.align();
                    return;
                }
                1 => {
                    let id = bits.vbr(8);
                    let inner = bits.vbr(4) as u32;
                    bits.align();
                    let len = bits.take(32) as usize;
                    let body = bits.pos + len * 32;
                    if id == 14 {
                        walk(bits, inner, body, true, names);
                    } else {
                        bits.pos = body;
                    }
                }
                3 => {
                    let rec = bits.vbr(6);
                    let n = bits.vbr(6) as usize;
                    let ops: Vec<u64> = (0..n).map(|_| bits.vbr(6)).collect();
                    if symbols && rec == 1 && ops.len() > 1 {
                        names.push(ops[1..].iter().map(|c| char::from(*c as u8)).collect());
                    }
                }
                other => panic!("abbreviation {other} in a module written without them"),
            }
        }
    }
}
