# gfx936 FlashAttention migration

Goal: port the CUDA r64c32d64 forward kernel to a small, standalone HIP
implementation and compare it with PyTorch SDPA on BW1000.

Scope: contiguous FP16 [B,H,N,64] Q and [B,H,M,64] K/V, noncausal,
dropout=0, forward only. N and M may have partial tiles. Default scale=1/8.
The user will compile and benchmark on BW1000; this workspace has no DTK.

Design: four wave64s cover 64 query rows. Each wave computes 16x32 scores
using 16x16x16 MMAC. For physical column j, read key perm(j), with
perm(j)=4*(j%4)+j/4, independently inside each 16-key tile. C lane t,
register r then describes original key 4*(t/16)+r, exactly the PV A layout.
V stays in its original logical order. FP32 online softmax/accumulators;
only P is rounded to FP16. Coalesced K/V loads prefetch through registers
and feed two LDS buffers. Transpose V while staging to make its B fragment
a contiguous four-half LDS read. No P shuffle or P LDS round trip.

- [x] Validate the lane-layout argument against dense attention on CPU;
      write BW1000 numerical tests for tails, cross-attention, scaling,
      large logits, uniform logits, invalid inputs and nondefault streams.
- [x] Implement flash_attn_bw.hip and a lightweight ctypes Python loader;
      build with hipcc/aicc --offload-arch=gfx936, no TileLang dependency.
- [x] Implement bench.py: correctness before timing, warmup, event timing
      of graph replay, auto/flash/math SDPA selection, actual SDPA operator
      reporting, median latency, speedup, TFLOP/s and JSON output.
- [x] Review implementation and run available CPU checks. Document exact
      target commands and distinguish local checks from pending GPU tests.

Local result: two CPU tests pass; four BW1000 test methods skip. Python
syntax and benchmark CLI checks pass. Building reports no DTK compiler,
so device compilation, device correctness and performance remain unverified.
An independent review agent was attempted twice but failed to return due
to tool stream disconnections; the implementation was reviewed locally.

Follow-up after user BW1000 results: original device compilation and
FP32-reference checks succeeded, but DTK Flash SDPA returned NaNs for
nonpositive scales. Those test cases now explicitly select SDPA Math;
positive-scale benchmarks still use the same SDPA backend. Original
long-sequence performance trails SDPA. CPU examination identified V LDS
bank aliasing; a regression test using the production header failed with
8 distinct words/bank for stores and 2 for reads, then passed at 1/1 after
folding high row bits into the V swizzle. Header changes invalidate the
build cache. Three CPU tests pass; the revised device code needs user
retesting. No improved GPU timing is claimed.

Second user result: all seven v2 tests passed on BW1000. Latencies for
N=256/512/1024/2048/4096 are 14.83/34.53/107.28/339.43/1186.11 us;
Flash SDPA remains faster from N=512 onward. Next controlled candidate
changes only exp2 call sites to __builtin_amdgcn_exp2f. The local HCU FA
reference explicitly requests native transcendental lowering, which the
v2 build did not. FA_BW_NATIVE_EXP2=0 restores generic exp2f for comparison;
benchmark metadata records the compiled selection. A new device test
checks 129 gradual online-max updates and reverse key ordering. Three
CPU tests pass, five device tests await the user's BW1000 run. Do not
attribute the remaining bottleneck to exp2 without the paired results.

Third user result: all eight v3 tests passed. Native vs libm at N=4096:
1015.19 vs 1186.19 us (Flash SDPA 659.37 us); native exp2 is retained.
Fourth candidate adds compile-time FA_BW_KV_TILE=64, leaving 32 as default.
It halves online updates/barriers at M divisible by 64, increases LDS from
16 to 32 KiB, and grows register fragments. Reuse the same 16-key column
permutation and expose compiled tile size in benchmark metadata. A failing
CPU test showed that using the ordinary stride-64 swizzle for V produces
four distinct words per bank on stores; the dedicated V swizzle restores
one for both tile sizes. CPU lane proof now covers both sizes and tails;
GPU tests cover 63/64/65/129 keys. compare_tiles.sh runs tests before timing
each configuration in fresh processes. All available CPU checks pass;
device compilation, correctness, register pressure and speed need user
measurements. Do not select tile=64 as the default before evidence.

Fourth user result: both tile sizes pass all eight tests, but C64 is slower
at every tested N. At N=4096: C32 1015.96 us, C64 1360.13 us. Retain C32.
No register spill or occupancy explanation is asserted without evidence.
Added diagnose.py to build both configurations with saved temporaries and
kernel-resource-usage remarks, preserving raw assembly/logs and a summary.
The summary parser counts only the target kernel and reports absent labels
explicitly; its two CPU tests pass. No kernel behavior changed in this
diagnostic step. Target compiler reports are still required for the next
performance decision.

User resource report: C32 has 96 VGPRs, 29 SGPRs, 16 KiB LDS and compiler
occupancy 2 waves/SIMD; C64 has 135 VGPRs, 31 SGPRs, 32 KiB LDS and
occupancy 1. Both have zero scratch and zero spills. Fifth candidate keeps
C32 and native exp2, moving next-tile global loads from before QK to after
softmax to reduce overlapping live ranges. A P-dependent empty asm memory
barrier prevents hoisting those loads back before softmax. Stage-to-LDS
timing and block barriers remain unchanged. FA_BW_LATE_PREFETCH=0 keeps the
verified baseline, =1 selects the candidate. compare_prefetch.sh compiles
both with resource remarks, tests before timing, and records the compiled
prefetch selection. Six local CPU/parser tests pass; five device methods
await BW1000. Whether reduced live ranges outweigh shorter load/compute
overlap remains unverified. Compiler version URL credentials are redacted
in future diagnostic reports.

User confirms hipprof on BW1000 and supplies DTK 26.04 help. Supported
interfaces used: --hip-trace --output-type 0 and --pmc --pmc-type 3.
Added profile_workload.py (fixed shape, CPU-generated inputs, eager calls,
no reference/Graph/nested profiler) and profile_hipprof.sh. The script
checks custom variants outside profiling, then collects before_qk,
after_softmax, and forced Flash SDPA into separate trace/PMC outputs.
Trace includes 5 warmup + 20 calls; PMC uses 3 calls and no warmup. Do not
interpret Python dispatch gaps as device stalls or PMC-instrumented time
as benchmark speed. Local syntax/CLI checks pass; actual collection and
available PMC fields remain for the user's target run.

User v5 and hipprof logs: both variants pass all eight tests. Delayed
prefetch reduces VGPRs 96 -> 91 but leaves occupancy at 2 waves/SIMD;
N=4096 regresses 990.78 -> 1049.56 us. Keep before_qk. HIPOPS reports 25
calls each, averaging 996.426 / 1060.816 / 663.533 us for before_qk /
after_softmax / Flash SDPA, consistent with the graph comparison. Host
allocation/startup statistics do not explain the steady device gap. The
provided PMC output contains only process-completion logs; no actual
counter values were supplied. Request existing run-8ynTvp artifacts,
without rerunning collection or speculating on the next device change.
The collector now archives each completed run automatically.

User cannot conveniently download reports. Added stdlib-only
dump_hipprof.py to preview existing PMC CSV and read-only SQLite data as
pasteable text, including file inventory, schemas and bounded sample rows.
It does not infer units/counter semantics or equate missing data with zero.
Real temporary CSV/SQLite smoke checks verified quoted fields, table names,
numeric values, empty-table reporting and unchanged database bytes. Ask
for terminal output from the existing run; no recollection or file download
is required.

User PMC CSV: C32 early/late both execute 15,843,328 LDS instructions and
78,213,120 VALU instructions, versus Flash SDPA 1,581,056 / 30,524,416.
Reported LDS bank conflicts and scratch are zero for all three. SDPA uses
232 VGPRs and 256 blocks versus our 96 VGPRs and 1024 blocks. Its template
parameters and dispatch suggest 256 query rows/block, 64 keys/iteration;
the source has not been inspected. Late prefetch reduces the aggregate
LDS wait counter but is slower. Neither register count nor wait counters
alone predict performance; VMEM instruction counts do not measure bytes.

Next experiment: FA_BW_Q_TILE=64/128/256, fixed four wave64s and C32.
Each wave owns 1/2/4 independent 16-row query parts. Load each K/V MMAC
fragment once and reuse it across parts. Keep Q and output staging in
64-row chunks with barriers before scratch reuse, so LDS stays 16 KiB.
Preserve the P register layout, FP32 softmax, early prefetch and native
exp2. Expose q_tile in compiled-library metadata. Keep Q64 default until
target correctness/resources/performance establish a better choice.

- [x] Add GPU checks for 127/128/129 and 255/256/257 rows, independent
      softmax state across query parts, and output guards.
- [x] Implement multi-part QK/PV reuse, constant LDS, metadata, and a
      compare_query_tiles.sh script with compiler resource remarks.
- [x] Review synchronization/indices, run local checks, document measured
      PMC evidence and exact target command. GPU validation remains remote.

Implemented the experiment. The many-update softmax test now uses 257
queries with different slopes, covering all query parts and a second
block. Local verification: six CPU/layout/parser tests pass; six GPU
methods skip because no DTK/BW1000 is available. Python syntax, shell
syntax and benchmark/workload CLI checks pass. Independent read-only
review found no issues. No target compilation, register allocation,
spill, GPU correctness or performance claims are made for Q128/Q256.

User v6 results: Q64/Q128/Q256 with C32 each pass all nine tests.
VGPRs=97/140/211, SGPRs=32/46/50, occupancy=2/1/1 waves/SIMD;
all use 16 KiB LDS with zero scratch/spills. N=4096 times are
988.01/846.50/888.01 us, versus Flash SDPA approximately 659 us.
Q128 saves 14.3% time versus Q64 but still trails SDPA (0.779x).
Q64 wins N256/512, Q128 N1024/4096, Q256 N2048. No universal best
tile or shape-generic dispatch policy is inferred from B1/H16 data.

Next controlled experiment uses existing kernel support, with no HIP or
Python API changes: compare Q128/Q256 x C32/C64. The earlier Q64/C64
failure accompanied occupancy dropping 2 -> 1, whereas both larger Q
tiles already have occupancy 1 with C32. C64 halves online updates and
barriers for full tiles but doubles LDS and increases registers, so
resource reports and numerical validation remain necessary, especially
for Q256/C64. Added compare_reuse_tiles.sh: four configurations, compiler
resource remarks, full tests before timing, forced Flash SDPA by default,
and separate results_v7_q{rows}_c{keys}.json files. Defaults stay Q64/C32.
Target compilation, correctness and performance of the new combinations
remain for the user's BW1000 run.

User v7 results: all four combinations pass nine tests. At N4096,
Q128/C32=846.80 us, Q128/C64=878.44, Q256/C32=887.82,
Q256/C64=914.27; Flash SDPA ~659.5 us. Q128/C64 uses188 VGPRs,
32KiB LDS, no spills, occupancy1. Q256/C64 uses256 VGPRs with
72 bytes scratch/lane and17 VGPR spills. C64 helps some smaller
shapes but does not close the long-sequence gap; keep C32 baseline.

Next change targets V staging rather than larger tiles. Found a separate
ordinary-LDS path in tilelang_ds_read_format.hpp:
ds_read_format_tensor_b_linear (lines306-348) uses
__builtin_hcu_ds_read_m32x16_f16. This was missed when checking only MLS
atom specializations. Its request lane t reads8 halves at [t/4,(t%4)*8]
in a16K x32N panel, returning B fragments [4*(t/16)+r,t%16+16*fragment].
Reference tests document the mapping, but target gfx938/946; gfx936 must
be checked directly. Preserve existing P layout and MMAC arithmetic.

- [x] Add production V address map preserving8-half vectors, CPU coverage
      of its bijection/address phases, and a target-only tagged LDS probe.
- [x] Add FA_BW_MATRIX_V=0(default)/1, vector V staging and two B fragments
      per matrix read; keep Q128/C32 fixed in the paired comparison script.
- [x] Record compiled V-load selection, review, run available checks, and
      document target compilation/layout/correctness/performance limits.

Implementation and independent review complete with no review findings.
Six CPU/layout/parser tests pass; seven device methods skip locally.
The new CPU phase test also rejects an unswizzled-address mutation.
Python and shell syntax checks pass. compare_matrix_v.sh runs paired
Q128/C32 configurations, including the16384-tag device probe only when
matrix reads are enabled. The probe shares production matrix_v.h; new
header changes invalidate the library build cache. Actual gfx936 builtin
compilation, lane mapping, full attention correctness and performance
are explicitly pending target testing. Default remains MATRIX_V=0.

User v8: native matrix V compiles on gfx936, with128 VGPRs (was140),
occupancy2 (was1),16KiB LDS, no scratch/spill. All full-attention numerical
tests pass, but the isolated tag probe matches only2048/16384: every
register shows that lane's first half. Script stops before native timing.
Host Clang minimal reproduction matches this exactly: scalar
__builtin_bit_cast(uint16_t, vector[r]) ignores the element index.
Whole-vector bit_cast followed by indexing preserves all8 elements.
Fix only the probe: whole-vector bit_cast +128-bit store. Preserve every
expected tag and failure condition; do not alter the kernel or relax tests.
Run the fixed target probe and attention tests before collecting native
timing. The compiler/occupancy report alone is not performance evidence.
Local O3 host reproduction over16384 values: old per-element bit_cast has
14336 mismatches; the exact new whole-vector store has zero. Target
confirmation remains pending; no broader unchanged tests were repeated.

User confirms v8 matrix V passes all10 tests and all16384 exact tags.
Q128/C32 native times for N1024/2048/4096 are74.94/246.49/796.87 us,
Flash SDPA68.74/192.87/659.36; v8 transpose controls80.36/262.49/846.58.
The long-case reduction is5.9%; native still takes20.9% more than SDPA.
Native is validated for the tested Q128/C32 configuration. Keep the global
Q64/transpose default pending evidence for its performance; do not invent
a shape-generic dispatch from this data.

Next step is current-kernel evidence, not another tile sweep: original PMC
was Q64/transpose, so collect Q128/transpose vs Q128/matrix vs SDPA.
profile_hipprof.sh now accepts third argument matrix (prefetch is legacy
default). Matrix/all also saves compiler assembly/opcode summaries outside
hipprof, then collects trace/PMC and automatically dumps selected CSV
fields for pasting. Added optional --columns regex to dump_hipprof.py;
unfiltered behavior remains, malformed widths retain raw rows, missing
columns report the header. Local shell/Python syntax and real temporary
CSV smoke checks pass. No kernel or ABI changes; GPU profiling remains
for the user. No claim of current bottleneck without the new evidence.

Current Q128 PMC confirms matrix V reduces LDS9,027,584 ->6,144,000,
VALU60,217,344 ->57,251,840; VMEM reads540,672 unchanged. SDPA reports
LDS1,581,056 and VALU30,524,416. Bank conflicts/scratch all zero.
Trace means transpose/matrix/SDPA:853.828/801.721/663.744 us.
The matrix version's LDS count matches2048 waves*(128 iterations*23+56):
per iteration8 K reads +4 matrix V reads +3 staging stores +8 softmax
shuffles. Shuffles alone account2,097,152 instructions. No faster gfx936
cross-row reduction builtin has been established; do not assume RDNA
permlane instructions apply. Counts do not establish cycle bottlenecks.

Next single-variable candidate: FA_BW_FULL_KV=0(default)/1. Compile an
additional FullKV=true kernel and dispatch only when m%KV_TILE==0.
Remove per-lane K/V load bounds and score masks in that kernel; preserve
generic false kernel for tails. No changes to softmax/P layout/rounding,
prefetch/V layout, Q/output masks or stream handling. Fix Q128/C32/matrix
V for paired tests. Expose enabled specialization in compiled metadata.
Update assembly diagnostics to report both template instantiations.
- [x] Cover full and tail M for scale/stability and partial Q tiles.
- [x] Implement specialization/dispatch, metadata and paired script.
- [x] Check multi-kernel assembly parsing, review and local verification;
      leave target compilation/correctness/performance for BW1000.

Local results: seven CPU/parser checks pass; seven device methods skip.
Python/shell syntax checks pass. Independent review found no actionable
issues. Attempting a generic Clang HIP host-only check with vendored HIP
headers was blocked by missing device math declarations; this is not a
DTK compile result and no target-build success is claimed. The paired
script tests each configuration before its own timing and reports both
specialization bodies. FA_BW_FULL_KV remains disabled by default pending
actual gfx936 compilation, numerical checks, resources and timing.

User v9: both configs pass all10 tests. FullKV removes conditional-select
instructions18 ->2 in static assembly, reduces VGPR128 ->124 and SGPR46
->34; LDS16KiB/occupancy2/no spills unchanged. Times generic/full:
N1024 75.06/68.12, N2048 246.64/221.41, N4096 797.15/712.71 us.
SDPA is68.84/192.97/659.14 for the full run. Long-case improvement10.6%,
remaining time gap8.1%; N1024 is approximately parity, not a broad win.

Next candidate: skip correction exp2, denominator rescaling and16 output
multiplies per query part when its online maximum does not increase.
Current correction then equals1. Keep the original FMA expression on
changed rows and plain denominator addition on unchanged rows. Put all
shuffles/MMACs outside the divergent branch. FA_BW_CONDITIONAL_RESCALE=0
remains the control; paired tests fix Q128/C32/matrix V/FullKV/native exp2.
Cost/benefit is uncertain because row-dependent branches can diverge.
- [x] Extend long softmax test with alternating query slopes so each wave
      mixes rows updating every tile and rows keeping their first maximum.
- [x] Implement conditional correction/accumulation and compiled metadata.
- [x] Add paired comparison script, review and verify locally; target
      resources/correctness/speed remain for the user's BW1000 run.

Seven local CPU/parser tests pass and seven GPU methods skip; Python and
shell syntax pass. Independent review found no correctness issue. A CPU
random-input model (not target measurements) observed5.23 maximum updates
per query across128 tiles;16-query wave parts had any update in38.5 tiles
on average. This motivates the experiment but does not predict its speed.
CONDITIONAL_RESCALE remains disabled by default until target tests and
timings establish the result.

User v10: both rescale variants pass all10 tests and16384 matrix-V tags.
Unconditional/conditional times for N1024/2048/4096:
68.84/72.69,222.75/227.75,716.95/716.70 us. FullKV=true VGPR123 ->126,
SGPR34, occupancy2, LDS16KiB, no scratch/spill. No measurable long-case
gain and shorter cases regress; retain CONDITIONAL_RESCALE=0. Unchanged
occupancy does not explain this regression.

Next experiment reuses existing kernel capabilities: Q64/128/256 x C32/64,
with MATRIX_V=1, FULL_KV=1, CONDITIONAL_RESCALE=0, native exp2 and early
prefetch fixed. Earlier tile sweeps used transposed V/generic KV; lower
register use may change rankings and spill behavior. Defaults remain
unchanged until combined configurations, including short shapes, are
measured. No automatic shape policy is inferred from a single B1/H16 run.
- [x] Add compare_tuned_tiles.sh: six configurations, complete tests before
      each benchmark, compiler resource remarks, forced Flash SDPA, and
      all five default lengths256/512/1024/2048/4096.
- [x] Save six JSON reports in a fresh run directory; summarize measured
      latency for each shape and the fastest sample's paired SDPA result.
- [x] Verify script syntax and summary behavior locally; new target
      compilation/correctness/resources/timing remain for BW1000.

Local verification: bash syntax passes. A temporary synthetic workflow
executes the actual shell script and summary, verifying all six fixed
configurations, tests preceding timing, forwarded shapes, forced Flash,
paired SDPA selection, separate run directories, failure short-circuiting,
and help without running tests. Synthetic times are not GPU measurements.
No kernel/API/default changes; unchanged GPU tests were not rerun locally.

User v11: all six Q64/128/256 x C32/64 configurations pass all10 tests
and all16384 tags. FullKV VGPRs are82/108/123/153/196/246, occupancy
3/2/2/1/1/1 waves/SIMD, LDS16/32KiB for C32/C64. No scratch or spills
in either full or generic kernels; Q256/C64 no longer spills.
Best measured B1/H16 configs for N256/512/1024/2048/4096:
Q64/C64=10.71/23.05us, Q128/C64=61.56us, Q256/C64=201.27us,
Q128/C64=704.09us. Paired Flash speedups2.167/1.363/1.119/0.962/0.937.
Long cases still take3.9%/6.8% more than SDPA with per-shape tile selection.

Consolidate verified settings: default Q128/C64, MATRIX_V=1, FULL_KV=1;
native exp2, early prefetch and unconditional rescaling unchanged. This
fixed tile beats the former Q128/C32 control at all five measured sizes;
it is not a universal best tile. Default N2048 time214.76us, not the
Q256/C64 winning time201.27us. Preserve explicit compile-time selection,
no automatic shape policy. Only default macros change; no arithmetic,
layout, synchronization, API, or tolerance changes.

Historical comparison scripts now pin old Q/V/FullKV settings explicitly,
so default promotion does not silently change their experiments. Add
profile_hipprof.sh all4096 tuned: fixed Q128/C64/matrix/full/native/early,
vs forced Flash only. Validate before profiling, save assembly summaries
for both KV template bodies, collect25 trace/3 PMC calls, print counters.
Legacy prefetch/matrix modes retain original full-KV-disabled controls.
Target performance already established by v11; current-configuration
PMC remains pending, and old Q128/C32/generic PMC cannot establish the
remaining bottleneck.

Local checks: seven CPU/layout/parser tests pass, seven device methods
skip without BW1000. All shell scripts pass syntax checks. Host-only
preprocessing confirms the old source with v11 explicit Q128/C64 flags
and the new default source expand identically; this is not a DTK build.
A temporary simulated profiler workflow exercises all nine mode/pair
combinations, verifies flags, collection counts, selected-tile diagnostics,
SDPA selection, report dump, and failure before collection. Independent
review identified historical README commands inheriting new defaults;
those examples now pin their original configuration or use fixed scripts.

User tuned-profile run: all10 GPU checks pass, FullKV kernel remains
153VGPR/35SGPR/32KiB/occupancy1/no scratch or spills. Static MMAC64,
exp34, bpermute8 and matrix-V reads8; no new runtime-counter conclusion.
Both custom and SDPA traces report "has not valid trace data"; both PMC
runs report "merge pmc data failed". No workload metadata or completion
output. Only empty trace DBs/logs are present. Unlike earlier successful
profiles, there are no kernel CSVs or PMC exports. Exact environment/
application/profiler failure remains unknown; do not change the kernel
based on missing counters or blame the compiler/GPU without evidence.

Confirmed local orchestration defect: relying on hipprof's exit status
lets reported failures return success and archives invalid data. Added
test_hipprof.py; a fake external profiler hides application output and
returns0 even on child/export failure. Original real shell script wrongly
returns0 in all five failure scenarios. New profile_capture.py executes
the existing workload via runpy in the same process, writes native/Python
stdout/stderr to a separate log and records status (missing on abrupt
exit). profile_hipprof.sh pins Python's absolute path, adds one eager
unprofiled preflight per variant, validates workload completion, known
hipprof errors and kernel CSV data rows. Stops on first failure, printing
log directory up front. Kernel and profile_workload.py are unchanged.

The first six new local regression tests and shell syntax pass. These tests
exercise real shell/capture/check code with external GPU/profiler fixtures;
they do not establish a fix for the target environment. Next target
command: bash profile_hipprof.sh trace4096 tuned after copying the shell
and new capture helper. Its own app log/status should distinguish failure
before/inside the profiled process from successful execution plus failed
export. Keep PMC separate until trace diagnosis succeeds.

Independent review reproduced two further false-success cases: injected
profiler errors only in workload.log, and API-only pmc.hipapi.csv matching
the broad export glob. Added two failing integration cases, then included
both logs in error checks and required the observed DTK pmc.csv filename
for PMC output (trace still requires a kernel CSV).
Final verification: all eight collection regression tests pass, and the
updated shell passes bash syntax checking. No target recovery is claimed.

Next target run stops in unprofiled preflight: capture started, exit0,
no workload metadata or completion. This narrows failure to workload
execution before hipprof invocation. Local production capture with the
real workload --help reaches argparse successfully (no GPU work). Empty
or main-definition-only temporary workloads reproduce the exact silent
exit0 pattern. Actual target source is not available, so empty/truncated/
missing entry is a hypothesis, not a confirmed cause. Recommend resyncing
the complete profile_workload.py (3387 bytes) and rerunning trace; keep
the completion guard and all device code unchanged.

Target trace recovery confirmed in run-t8YSjD: all10 tests pass, both
preflights complete1 call, both profiled workloads complete25 calls,
status and kernel CSV checks pass. Custom Q128/C64/matrix/full/native/
early/unconditional mean17714565ns/25=708.5826us; Flash SDPA
16548174ns/25=661.92696us. Time gap7.048%, speedup0.9342x, consistent
with graph timing. Trace includes5 warmups; CPU HIP API times do not
replace HIPOPS device durations. Exact old target-file defect remains
unverified; recovery does not prove it was empty or missing main.
No new code changes. Next: bash profile_hipprof.sh pmc4096 tuned;3 calls
per implementation and automatic CSV preview. Current counters still
required before selecting another kernel optimization.

Current PMC now succeeded in run-cFbzCO,3 calls each. Q128/C64 vs SDPA:
LDS4571136/1581056, VALU43280384/30524416, VMEM_RD540672/401408,
VMEM_WR16384/69632, bank-conflict and scratch zero. LDS waits about
2.93M/1.30M; counts are not cycle fractions. PMC reports allocated
VGPR156/SGPR48 versus compiler153/35, not evidence of a new regression.

Next bounded candidate: default-off FA_BW_ASYNC_V, complete-KV only,
matrix V and early prefetch required. hcu/copy.h and core/arch/
amd_buffer_addressing.hpp explicitly include gfx936 for16B buffer-to-LDS.
Use the reference raw buffer descriptor format0x20000 with tile-sized
extent, wave LDS base via m0, and inverse V swizzle on global indices.
Keep K through registers and P/MMAC/math unchanged. Issue next V to the
alternate buffer before QK; wait vmcnt0/lgkmcnt0 before the existing
end-of-iteration block barrier. First tile also waits. No pending direct
loads on final iteration. Generic tails retain the original path.

- [x] Add CPU source-to-destination mapping proof (initial compile fails
      before helper exists) and production async_v_source helper.
- [x] Add direct-to-LDS helper, full-only dispatch inside template, compiled
      metadata, header cache dependency and paired Q128/C64 script.
- [x] Extend exact-tag device probe for32/64 keys and16B input offsets;
      cover4096 full and4097 tail many-update softmax/pipeline behavior.
- [x] Review descriptor/layout/wait correctness; complete local validation.
Target instruction compilation, tags, synchronization and speed remain
unverified. Do not select the candidate by instruction-count conjecture.

Seven local CPU/layout/parser checks pass; eight device methods skip.
Python and comparison-script syntax pass. A temporary source mutation
removing the inverse swizzle fails the production layout test as intended.
Independent source review found no blocking descriptor/layout/wait issue.
The paired script validates the full C64 pipeline; C32 has a standalone
copy/fragment probe, but its full async attention pipeline needs a separate
KV_TILE=32 run before any future promotion. ASYNC_V remains default-off.

User v12 control:10 pass/1 expected skip; Q128/C64 N1024/2048/4096
62.29/215.10/704.47us. Async candidate fails during C++ compilation:
readfirstlane returns int, rejected as narrowing inside uint32 vector
initializer. No candidate device correctness or timing collected.
Fix only the two returned address components with explicit uint32_t
casts. Real local Clang AMDGPU builtin reproduces the old error; actual
production header now type-checks for both32/64 templates. This used
gfx90a syntax-only and shim HIP/type declarations, not DTK/gfx936
assembly or execution. Update async_v.h and rerun only the candidate
tests/benchmark; retain all existing checks and default-off selection.

User async V result after narrowing fix: all11 tests and both8192/16384
async tags pass. Full kernel145VGPR/35SGPR/occupancy1/32KiB/no spills;
generic unchanged160VGPR. N1024/2048/4096=60.86/212.07/695.16us vs
control62.29/215.10/704.47us, reductions2.3%/1.4%/1.3%. Paired Flash
68.93/192.96/659.20us. No general win; occupancy remains unchanged.

DTK emits reserved-m0 clobber warning. Change only async_v.h: save m0 to
an early-clobber SGPR output, set m0/load/restore within one asm, remove
m0 clobber while retaining memory. Generic Clang AMDGPU assembly
generation with -Werror=inline-asm reproduces old warning as error;
updated production header with -O3 emits all three C32/C64 copy sequences
without warning, saving/restoring to distinct SGPRs from all input regs.
No target compile/run claim from this gfx90a/minimal-declaration check.
Extra scalar moves may change timing; rerun candidate to separate the
new m0 preservation code from the measured695.16us version. Keep default
ASYNC_V=0 pending that evidence.

User confirms m0-preserving version passes target compilation with
-Werror=inline-asm, all11 tests, and all async/matrix tags. FullKV
145VGPR/35SGPR/32KiB/occupancy1/no scratch/spill remains. New custom
N1024/2048/4096=60.96/213.00/698.82us, paired Flash68.86/193.57/659.06us,
speedup1.130/0.909/0.943. Versus earlier register-V controls, reductions
2.1%/1.0%/0.8%; those small differences are not a fresh paired comparison.
N4096 still takes6.0% more time than Flash. Retain the now target-validated
async V implementation as optional, default-off; no further kernel changes
in this result-review step. Migration/correctness is established, but no
claim of universally outperforming Flash SDPA or stable sub-1% speedup.

## Async K candidate: preserve K layout, remove intermediate staging

User asks how to continue optimization toward beating SDPA. Best measured
Q256/C64 at N2048 is201.27us vs193.65us; current Q128/C64 async V at
N4096 is698.82us vs659.06us. Try direct K at both query sizes, with V
direct enabled in controls as well. No new default or math change.

- [x] Add failing CPU mapping test before implementation, plus device
      exact-tag/guard probe for C32/C64 and its unittest entry.
- [x] Add default-off async_k.h: x2 direct load, inverse K XOR source,
      tile-sized descriptor, explicit unsigned casts and m0 preservation.
- [x] Integrate only FullKV; skip register K loads/stores, issue next
      K before QK, share validated async V wait/barrier. Tail unchanged.
- [x] Add loader/cache dependency, metadata and Q128/Q256 paired script.
- [x] Verify locally and obtain independent read-only review.

Local7 CPU/layout/parser checks pass,9 GPU methods skip. Removing the
inverse K swizzle in a temporary copy makes the layout test fail.
Generic AMDGPU -O3 -S -Werror=inline-asm emits6 x2 copies across C32/C64,
with m0 saved/restored to SGPRs distinct from address/resource inputs.
Host-only syntax checks instantiate full/tail templates in18 configurations
using minimal HIP/type/math declarations. Neither check is gfx936 code
generation/execution. Independent review found no blocking defect.

Target next: bash compare_async_k.sh --lengths 1024 2048 4096
Require all tags/guards and attention checks before timing; inspect spills
and compare each K1 candidate against its K0/V1 control and paired Flash.
The integrated comparison is C64 only; C32 needs a separate full-pipeline
target test before any promotion. Async K/V remain default-off. K x2
increases load instruction count versus register-prefetched x4, so speedup
is an experiment, not an inference from fewer LDS stores.

Target async K results: all four configs compile with -Werror=inline-asm,
K0 each11 pass/1 expected skip; K1 each12 pass and all K/V tags/guards match.
Q128 K0/K1 us:61.07/60.46,213.06/211.08,698.93/690.07. VGPR145->147,
SGPR35->44. Q256:66.90/67.28,199.79/199.33,725.95/724.42; VGPR239->232,
SGPR40->44. All32KiB/occupancy1/no scratch/spill. Gains are small; Q256
long-sequence0.2% changes are not confirmed stable improvements. Best
N2048=199.33 vs paired193.38us (+3.08% time), N4096=690.07 vs658.94us
(+4.72%). Keep async defaults off; integrated C64 correctness established.

## Defer denominator reduction: remove two row shuffles per KV iteration

Before interleaving QK/PV, remove repeated denominator communication.
All four lanes for each query share a reduced max and thus every alpha.
Keep local d_l = alpha*d_l + local_probability_sum; reduce once before
output. Real-arithmetic equivalence follows by linearity; FP32 addition
order differs, so retain existing tolerance and full target verification.
Maximum reduction, FP16 P, FP32 accumulation and memory pipeline unchanged.

- [x] Add failing CPU proof with unequal lane sums; extend model and add
      target group-mass check for N129 and M64/128/4096/4097, multiple heads.
- [x] Add default-off FA_BW_DEFER_DENOM, final row reduction, ABI/metadata.
- [x] Add paired Q128/Q256 script at C64, ASYNC_K=ASYNC_V=1, forced Flash.
- [x] Local8 CPU/layout/parser checks pass,10 GPU methods skip. Lane0*4
      mutation rejected. Minimal-declaration host syntax covers24 configs
      including full/tail templates and conditional rescale0/1; no target
      build or execution implied.
- [x] Independent read-only review found no blocking defect; confirmed
      lane grouping, tail behavior and conditional-rescale compatibility.
      Target comparison keeps conditional rescale at its default0.

Target next: bash compare_defer_denom.sh --lengths 1024 2048 4096
At M4096/C64, denominator shuffles per query part fall128->2; maximum
shuffles still run each tile. This source-level reduction is not a predicted
kernel speedup. New mode is default-off pending target checks and timings.

Target v14: all4 configs14 tests pass. Q128 defer0/1 us:
60.62/59.92,211.30/206.57,690.20/673.86. Full VGPR147->148,
tail160->171; no spills. Q256:67.13/66.15,199.32/195.97,724.41/712.19;
full232->256/no spill; tail247->256 with10VGPR spills/44B scratch per lane.
All32KiB/occupancy1. Best N2048=195.97 vs192.92us (+1.58% time),
N4096=673.86 vs660.11us (+2.08%). Restrict deferred denominator to FullKV
to restore original tail algorithm; full C64 math remains as measured.

## C128 extension requested by user

User steered next experiment to K128/C128. No Q192 code was added.
C128 halves KV iteration count versus C64; adds score/P registers and
raises double-buffer LDS to64KiB. Compare Q64/Q128 at C64/C128 with
async K/V and deferred denominator enabled. Keep default C64 and flags0.

- [x] Extend CPU mapping/matrix-bank/register-reuse cases and target
      probes to128keys; add M127/128/129/256 and C128 scale/stream cases.
- [x] Reproduce C128 rejection at kernel/helper static assertions, then
      allow128 in kernel and async helpers. Require matrix V for C128.
- [x] Keep chunk loops, swizzles and MMAC layouts; C128 uses16KiB
      descriptors,8 K x2 and4 V x4 requests per thread.
- [x] Test V128 at48KiB LDS offset with48KiB observable prefix guard,
      matching V1 at the upper end of64KiB storage.
- [x] Add compare_k128.sh, metadata for CU/shared-memory limit and
      full-only denominator scope; tighten index guard to2**31-129.
- [x] Local8 CPU/layout/parser checks pass/10GPU skips;36 host template
      configurations pass with minimal declarations. Generic AMDGPU -O3-S
      emits12 copy asm blocks with m0 preserved and no input SGPR overlap.
- [x] Read-only review confirms buffer bounds, descriptors, tags and
      fallback; follow-up fixes length overflow guard and checks high LDS.

Target next: bash compare_k128.sh --lengths 1024 2048 4096
Require GPU tests and64KiB launch success before timing. Compare both
tile sizes at each Q; inspect actual resource allocation/spill. No claim
of gfx936 compile, C128 correctness or speedup from local checks.

Target v15 C128 results received:4configs each14tests pass; all K/V tags,
64KiB high-buffer/prefix guards pass. Both full/tail kernels have0scratch/
spills. CU count80; missing shared_memory_per_block property does not
report a hardware limit. N1024/2048/4096 us:
Q64C64=73.73/233.51/818.35; Q64C128=94.10/296.35/1041.23.
Q128C64=59.24/206.23/674.03; Q128C128=69.80/246.59/808.48.
Full VGPR102/124/148/210, SGPR44; LDS32/64/32/64KiB; occupancy2/1/1/1.
C128 regresses27% atQ64 and18-20% atQ128, so keep C64 as optimization
basis. Do not attribute all regression to occupancy (Q128 unchanged).
Q128C64 tail restores160VGPR/no spill. Paired Flash atN4096=659.22us,
custom still+2.25% time. C128 remains supported/validated, not default.

## Profile the current measured Q128/C64 implementation

Prior tuned PMC predates async K/V and deferred denominator. Collect fresh
evidence before another kernel change. No kernel/default changes here.

- [x] Add optimized profile mode fixingQ128/C64, asyncK/V/defer1, matrixV/
      fullKV/nativeexp2/earlyprefetch; conditionalrescale0. Explicitly pin
      old modes to asyncK/V/defer0.
- [x] Add regression verifying actual workload flags in preflight/trace/PMC
      for optimized and tuned despite stale callerQ256/C128 environment.
      Red before mode exists; all9 profiling tests now pass; shell syntax
      passes. Independent read-only review found no issue.

Target next: bash profile_hipprof.sh all 4096 optimized
Get assembly summary plus current trace/PMC and forced Flash comparison.
Profile data remains target-pending; do not use old PMC as current counts.

Current optimized PMC received from run-Eshmnk: verified options,14 tests
pass, full148VGPR/44SGPR/32KiB/occupancy1/no spills. Trace25 calls includes
5warmups:680.570us vs659.482us (+3.20% time). Three PMC calls each:
LDS3,268,608 vs1,581,056; VALU41,590,784 vs30,524,416;
VMEM_RD802,816 vs401,408; VMEM_WR16,384 vs69,632;
WAIT_LDS622422..631004 vs1298526..1307272; bankconflict/scratch0.
Grid workitems131072/65536 atwgr256 =>512/256 blocks. Total instruction
counts include different query blocking. Wait counts are not cycle shares.
Current LDS wait counter is substantially lower than earlier optimized
baseline (~2.93M); investigate VALU/softmax dependency chains next.

## Four independent lane-local reduction chains

Default-off FA_BW_REDUCE4 applies onlyFullKV. For each register column r,
initialize max/sum from first16-key fragment, update across later fragments,
then combine four lanes-local values in a balanced tree. Scaled logits,
row shuffles, exp2, FP16 P, FP32 accumulation and memory pipeline unchanged.
Only FP32 sum association changes. Extra accumulators may increase VGPRs;
benchmark Q128 and Q256 to distinguish useful scheduling from regressions.

- [x] Add failing CPU model test before reduce4 model/implementation;
      cover uneven register probabilities, rising/falling maxima and tails.
- [x] Add kernel flag/scoped implementation, getter/ctypes/metadata and
      compare_reduce4.sh with fixed C64/asyncKV1/defer1, forced Flash.
- [x] Local9 CPU/layout/parser checks pass,10 GPU methods skip. Missing
      fourth sum accumulator mutation fails.36 host configs pass using
      minimal declarations; not DTK/gfx936 validation.
- [x] Independent read-only review found no issue with initialization,
      element coverage, tree reductions, scope or interfaces.

Target next: bash compare_reduce4.sh --lengths 1024 2048 4096
Require accuracy tests and inspect register/spill reports before accepting
timings. No target compile/correctness/speedup claim yet; defaults unchanged.

Target v16 results: all four configurations pass all15tests. Graph us at
N1024/2048/4096: Q128R0=59.26/206.30/673.77; R1=59.88/207.77/679.20;
Q256R0=66.14/195.97/712.23; R1=67.46/199.42/724.14.
Full Q128148->147VGPR,44SGPR,no spill,1256->1252static instructions.
Full Q256256->255VGPR,44->48SGPR,0->12B scratch/2VGPRspills,
2140->2151instructions. All32KiB/1wave per SIMD; tails160/247VGPR,no spill.
Reject REDUCE4 for performance; retain default0. Q128 regresses without
spilling too. Best paired gaps:Q256N2048+1.59%;Q128N4096+2.18% time.

## Q192 between measured Q128 and Q256

Support three64-row query parts within the existing four-wave kernel;
no new math/layout/copy algorithm. Keep Q128 default and all flags unchanged.
Fewer blocks thanQ128 reduce repeated K/V reads; fewer states thanQ256
may avoid its256VGPR limit. Actual resource allocation remains unknown.
AtB1H16N4096:512->352blocks, but padded rows4096->4224 (+3.125% math).

- [x] Observe pre-change host syntax rejection ofQ192.
- [x] Extend existing GPU output/guard tests to191/192/193,383/384/385;
      cover fullM128 and tailM97. They detect missing third query part,
      wrong launch coverage and staging overwrite, against independent FP32.
- [x] AllowQ192 in kernel; compare_q192.sh fixesC64/asyncKV1/defer1/reduce0
      and rerunsQ128/Q192/Q256 tests plus forcedFlash benchmarks.
- [x] Local9 CPU/layout/parser checks pass;10GPU methods skipped.36host
      template configurations pass (fourQ x threeC x threefeature sets;
      FullKV true/false instantiated). Comparison script shell syntax passes.
- [x] Independent read-only review: no power-of-two assumption inQ_PARTS;
      third query part, staging barriers and ceil launch coverage verified.
- [x] User-run DTK correctness/resources/timing forQ192 received below.

Target next: bash compare_q192.sh --lengths 1024 2048 4096
Do not infer a target speedup or spill-free compilation from host checks.

Target v17: threeconfigs each15tests pass. Q192 full189VGPR/48SGPR,
tail208VGPR/72SGPR, all0scratch/spill,32KiB,1wave/SIMD;
static instructions1713full/1830tail. N1024/2048/4096 us:
Q128=59.30/206.31/673.90; Q192=82.92/224.14/690.30;
Q256=66.05/195.99/712.14. RejectQ192 for these shapes.
Best paired Flash gaps:Q256N2048+1.33%,Q128N4096+2.25% time.

## Wave-uniform rescaling gate

Default-off FA_BW_WAVE_RESCALE, FullKV only, mutually exclusive with old
per-row CONDITIONAL_RESCALE. All64lanes call __any(new_max>old_max) before
the branch. If any row updates, entire wave executes original correction,
denominator and accumulator scaling. If none update, all corrections are
exactly1: only add tile_sum and retain accumulators until PV. No thresholds,
approximate max updates, changed MMAC/layout or new buffer synchronization.
Vote/branch overhead and register pressure may outweigh saved instructions.

- [x] Add failing CPU test before implementing wave gate/model option.
      None/all/onlylastrow changing cases compare against dense attention;
      add GPU tests withN257H2,M128/4096/4097.
- [x] Add kernel option, compiled getter, loader and metadata; comparison
      fixesC64/asyncKV1/defer1/reduce0 and testsQ128/Q256 x wave0/1.
- [x] Local10 CPU/layout/parser pass,11GPU methods skipped; incorrectlane0
      vote mutation causes8 numerical failures.48host syntax configs pass
      (FullKV true/false, fourQ,threeC, old/new rescale and denom modes).
      Python/shell syntax pass; noDTK/GPU claims.
- [x] Independent read-only code review found no blocking issue: wave vote
      is converged, first/mixed/unchanged cases and tail fallback are valid;
      hardware code generation/resources/performance remain target-pending.
- [x] Target: bash compare_wave_rescale.sh --lengths 1024 2048 4096

Inspect full/tail resource reports and require correctness before timings.

Target v18 received:4configs each17tests pass, no scratch/spill.
N1024/2048/4096 us:Q128W0=60.13/206.55/673.94;
Q128W1=65.89/231.52/753.10;Q256W0=66.08/195.82/712.25;
Q256W1=71.90/209.49/756.45. Full VGPR148->143 and256->247,
SGPR44 unchanged; LDS32KiB/occupancy1 unchanged. Static instructions
1256->1275 and2140->2180; tail resources unchanged.
Rejectwave-rescale, default0 remains. Regression9.6-12.1% atQ128,
6.2-8.8% atQ256. Best paired gaps1.61% at2048 and2.21% at4096.
Do not infer that ballots/branches alone explain loss without codegen/PMC.

## Inspect actual code generation before the next kernel change

No kernel/default changes. Add detailed, bounded, pasteable assembly output
to identify actual max/add/mul/FMA/branch instructions and read/wait/MMAC
ordering omitted by the previous selected-opcode summary.

- [x] Add two initially failing detail-parser tests; implement all-opcode
      counts, wait-operand counts and four bounded context anchors, merging
      overlaps and excluding comments/other functions. Missing anchors are
      reported; no inference that anchors identify runtime hot paths.
- [x] Add --details/--full-only and inspect_codegen.sh:Q128W0,Q128W1,Q256W0,
      C64/asyncKV1/defer1/reduce0. Missingfull/malformed/noassembly fails.
- [x] Six parser tests pass; controlled compiler fixture verifies actual
      script options despite stale env, stop-on-compile-failure and missing
      full-kernel failure. Python/shell syntax passes; notDTK validation.
- [x] Independent read-only review found and fixed handling of Clang ';'
      comments on labels/waitcnts; added real-style regression before fix.
      Six tests pass again; reviewer confirms no blocking issue, correct
      source line numbers and no static-to-runtime overclaim.
- [x] User collects: bash inspect_codegen.sh

Use the resulting instructions to choose the next change. No speedup claim.

Target codegen received: Q128W0 148VGPR, W1 143VGPR, Q256W0 256VGPR,
all no scratch/spill. Already max3/FMA/packed FP32 arithmetic inW0.
Q128W0/W1: fma38/6, pk_add16/0, add4/38, sub2/34;
lgkmcnt0 waits25/32 but total waitcnt49/49. W0 excerpts interleave QK
and softmax, and V reads with rescale. W1 excerpts show earlier rescale
branch and prompt wait0 after firstV read. Static evidence establishes
changed fusion/scheduling, not per-instruction runtime attribution.
W0 scales eachscore formax, then usesrawscore+scale again inprobabilityFMA.

## Delay max-path scaling until after reduction

Default-off FA_BW_DELAY_SCALE, onlyFullKV withpositive scale. Reduce raw
max, multiply one maximum bylog2_scale, then compare withold scaled max.
Explicit fmaf(rawscore,log2_scale,-newmax) produces shifted probabilities.
Forpositive scaling max(s*x)=s*max(x); C64 source scales16->1 perquerypart,
saving30 perlane periteration atQ128 or60 atQ256 before compiler scheduling.
Not a target instruction-count or speedup claim.

Host dispatch routes M tails and scale<=0 toexistingFullKVfalse whenflag1.
These fallback calls retain original memory copy and denominator reduction.
No new device-loop runtime branch; existing defaults/math types/layout stay.

- [x] Add failing CPU test, then model+kernel option, getter/metadata scope,
      and compare_delay_scale.sh withQ128/256 C64 asyncKV1 defer1 W0 R0.
- [x] Add GPUpositive scales/logitoffset tests and highmagnitude negative
      scale fallback case. Existingzero/tails/guards/currentstream retained.
- [x] CPU/layout/parser13 pass,12GPU methods skipped. Missingmaxscale
      mutation fails24cases.48host syntax configs pass, FullKVtrue/false;
      Python/shell syntax passes. NoDTK/GPU validation locally.
- [x] Independent read-only review found no blocking issue; positive-scale
      dispatch, scaled max units, FMA and fallback verified. Clarified FMA
      rounding can give a positive exponent residual; no clamp added.
- [x] Target: bash compare_delay_scale.sh --lengths 1024 2048 4096

Target v19 received: fourconfigs each19tests pass, including shifted logits,
high-magnitude negative-scale fallback, tails, tags, guards and streams.
N1024/2048/4096 us:Q128S0=59.20/206.32/673.93;
Q128S1=57.31/198.92/648.08;Q256S0=65.98/195.96/712.11;
Q256S1=64.66/188.99/683.71. Q128full148VGPR,static1256->1234;
Q256full256VGPR,static2140->2094;SGPR44,LDS32KiB,occupancy1 unchanged,
no scratch/spills;tail160/247VGPR,no spills.

Measured per-shape wins over paired Flash:Q128N1024=57.31vs68.91 (1.202x),
Q256N2048=188.99vs193.75 (1.025x),Q128N4096=648.08vs659.34 (1.017x).
Accept delay_scale in recommended preset. These are B1/H16/D64,noncausal,
scale.125,FP16,Graph results; not an all-shape or single-default claim.

## Preserve and document the measured winning configurations

No further kernel/default changes. Add bench_best.sh to run the exact
Q128N1024/4096 andQ256N2048 settings against forcedFlash, each run in a
fresh result directory. Existingbench validates outputs before timing.
Put current results/options prominently inREADME, correct current test
count and exp2 rounding note. Keep experimental flags available for history.

- [x] Record v19 timings, resource counts and validation scope.
- [x] Add fixed reproduction script and documented Python build flags.
- [x] Shell syntax and actual script option selection verified with a
      recording workload, including distinct directories across two runs.
      Independent read-only review confirms settings and result scope.
      No kernel edits or additional GPU performance claims in this step.

## Extend to B1 / H32 / N4096 / D128

Keep the QK column permutation and shared forward algorithm. Compile D64
and D128 separately, selected by the tensor's last dimension, with default
scale 1/sqrt(D). Preserve D64 defaults; use Q64/C32 initially for D128.
Compare Q64/Q128 x C32/C64 with async K/V, deferred denominator and delayed
scale. D128/C128 exceeds the current 64 KiB LDS limit and is rejected.

- [x] Add failing CPU tests for upper-channel copy addresses, D128 input
      alignment and mismatched dimensions; then generalize head strides,
      register fragments, matrix panels and async descriptors/copy counts.
- [x] Add dimension-specific library cache and benchmark/profile arguments.
- [x] Reuse the GPU regression suite at D128; add upper-channel/default-scale
      and same-process D64/D128 cases. Probes cover all four V panels and
      the highest V buffer, plus guarded async K copies.
- [x] Add compare_d128.sh with fixed B1/H32/D128, N4096 by default,
      forced Flash SDPA and fresh result directories with a printable table.
- [x] Local suite: 26 CPU tests passed, 26 GPU tests skipped. 18 kernel host
      syntax configurations passed, D128/C128 rejected as intended. Generic
      gfx90a assembly accepts D128 async copies with expected static counts
      (C32: 4 x2 K / 2 x4 V; C64: 8 x2 K / 4 x4 V).
      Script dispatch was checked with a recording-only workload, not timed.
- [x] Independent read-only review found no concrete regression.
- [ ] BW1000: run bash compare_d128.sh. gfx936 compilation, resource usage,
      device instruction semantics, HIP module isolation, numerical accuracy
      and performance remain unverified locally. No D128 speedup claim.


### First D128 target feedback: stale standalone probes

B1/H32/D128 Q64/C32: full kernel 127 VGPR / 52 SGPR, 32 KiB LDS,
compiler occupancy 2 waves/SIMD, 1168 static instructions; tail 142 VGPR /
46 SGPR, occupancy 1, 1219 instructions. No scratch or spills in either.
All 11 non-probe GPU tests passed, including upper channels, scales,
output guards, streams and mixed D64/D128 libraries; 4 CPU tests passed.
Three standalone probe errors stop the script before any timing.

Evidence of old probe sources paired with new headers: V expected rows
advance by 64 (8256, 8320, 8384) but actual by 128 (8320, 8448, 8576);
V C32 count is 8192 instead of 16384. K count is 4096 instead of 8192.
CPU address model of old D64 K reads against D128 storage reproduces all
8 logged first mismatches exactly. Both current .hip probes already use
HEAD_DIM for their allocations, addressing, host expectations and panels.

- [x] Add probe layout-version preflight before GPU build/run, and print
      actual D/C, LDS and expected tag counts in both standalone probes.
- [x] Verify unversioned probes rejected before adding markers; current
      probes accepted afterward. Mixed-directory checks reject either old
      probe independently. No kernel, memory layout or math change.
- [x] Local attention suites: 11 CPU checks passed, 26 GPU tests skipped;
      15 probe host syntax configurations passed (not device compilation).
      Independent read-only review found no blocking issue.
- [ ] Sync test_matrix_v.hip, test_async_k.hip and test_flash_attn.py on
      BW1000, then rerun compare_d128.sh with all correctness tests enabled.
      No performance measurements were obtained in this failed run.


### D128 Q64/C32 validated; stale benchmark blocks timing

User rerun build/d128/run-iupBVT: all 18 tests pass in 20.408s.
K C32/C64 tags+guards: 8192/16384 match; async V C32/C64:
16384/32768 match; matrix V C32: 16384 match, all at D128.
This confirms the probe-source synchronization diagnosis. Q64/C32
attention is validated, while the other three attention tile configurations
and all performance measurements remain pending.

Remote bench.py rejects --head-dim 128; its usage lists no such option.
The current local bench.py already supports D128 across allocation, library
selection, default scale, references, metadata and FLOPs. Sync bench.py
and compare_d128.sh. The latter now checks actual benchmark --help before
running tests or creating result directories, giving a direct stale-file error.


### D128 first four-tile timing and larger-Q follow-up

User graph timing summary, B1/H32/N=M4096/D128:
Q128/C32: 2031.34 vs Flash1521.38 us, 0.749x.
Q128/C64: 2471.82 vs Flash1521.49 us, 0.616x.
Q64/C32: 3755.25 vs Flash1521.67 us, 0.405x.
Q64/C64: 3745.27 vs Flash1521.07 us, 0.406x.
Q128/C32 is the measured winner; needs ~25.1% lower latency to tie Flash.
No resource counts for Q128 were provided with this timing-only summary.
The Q64->128 improvement motivates more K/V reuse, but does not establish
bandwidth or register pressure as the dominant bottleneck.

Add compare_d128_qtiles.sh: fix C32, compare Q128 baseline with existing
Q192/Q256 specializations, all at D128. Explicit dimension macro for direct
diagnose.py builds avoids accidentally collecting D64 resources. Each
configuration gets a fresh diagnostic compile (even if the callable library
is cached), full tests, paired Flash timing and isolated logs/JSON. No kernel
or default change. Q128/192/256 all use32KiB LDS; B1/H32/N4096 block counts
are1024/704/512. Q192/Q256 may spill and are unmeasured candidates.

- [x] Host syntax: D128/C32 Q128/Q192/Q256, full and masked kernels pass.
      D128 CPU suite4 passed,14GPU methods skipped (no local BW1000).
- [x] Recording-only workflow validates flags, dimension, three candidates,
      diagnose->tests->benchmark order, fresh directories across two runs,
      and immediate stop on diagnostic failure through tee/pipefail.
- [x] Independent review found no blocker. Q192 also computes 128 padded
      rows at N4096 (3.125%); no speedup inferred from reduced block count.
- [ ] Target: bash compare_d128_qtiles.sh; return timings plus compiler
      VGPR/scratch/spill information from the saved resource logs.
