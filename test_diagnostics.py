import unittest

from diagnose import redact_url_credentials, summarize_assembly, summarize_assemblies


class AssemblySummaryTests(unittest.TestCase):
    def test_details_keep_math_waits_and_instruction_context_in_target_kernel(self):
        source = """
unrelated:
    v_exp_f32 v99, v99
.Lfunc_end0:
_Z30flash_attn_v2_r64c32d64_kernelILb1EE:
    buffer_load_dwordx2 v0, s[0:3], 0 offen lds
    s_waitcnt vmcnt(0) lgkmcnt(0)
    v_mmac_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]
    v_max3_f32 v0, v1, v2, v3
    v_mul_f32 v0, s4, v0
    s_cbranch_scc0 .LBB1_4
    // v_exp_f32 v88, v88
.LBB1_4: ; =>This Inner Loop Header: Depth=1
    v_exp_f32 v0, v0
    v_add_f32 v1, v1, v0
    ds_read_m32x16_b16 v[4:7], v2
    s_waitcnt lgkmcnt(0) ; wait for matrix operands
    s_waitcnt lgkmcnt(0)
.Lfunc_end1:
other:
    v_exp_f32 v77, v77
.Lfunc_end2:
"""
        got = summarize_assembly(source, details=True)
        self.assertEqual(got["static_opcodes"]["v_max3_f32"], 1)
        self.assertEqual(got["static_opcodes"]["v_mul_f32"], 1)
        self.assertEqual(got["waitcnts"], {"s_waitcnt vmcnt(0) lgkmcnt(0)": 1,
                                          "s_waitcnt lgkmcnt(0)": 2})
        contexts = "\n".join(line for window in got["windows"] for line in window["lines"])
        self.assertIn("s_cbranch_scc0 .LBB1_4", contexts)
        self.assertIn(".LBB1_4:", contexts)
        self.assertIn("v_exp_f32 v0, v0", contexts)
        for wrong in ("v99", "v88", "v77"):
            self.assertNotIn(wrong, contexts)
        self.assertEqual(got["missing_anchors"], [])

    def test_details_bound_output_and_report_missing_instruction_family(self):
        symbol = "_Z30flash_attn_v2_r64c32d64_kernelILb1EE"
        body = ["    v_add_f32 v0, v0, v1"] * 240
        body[120] = "    v_exp_f32 v0, v0"
        source = symbol + ":\n" + "\n".join(body) + "\n.Lfunc_end0:\n"
        got = summarize_assemblies(source, details=True)[0]
        self.assertEqual(got["static_opcodes"]["v_add_f32"], 239)
        self.assertEqual(len(got["windows"]), 1)
        self.assertLessEqual(len(got["windows"][0]["lines"]), 41)
        self.assertIn("first_mmac", got["missing_anchors"])
        self.assertIn("first_matrix_read", got["missing_anchors"])
        self.assertIn("last_buffer_load", got["missing_anchors"])

    def test_compiler_version_removes_url_credentials(self):
        text = "clang (https://build-user:dummy-secret@example.invalid/compiler.git abc123)"
        self.assertEqual(redact_url_credentials(text),
                         "clang (https://[redacted]@example.invalid/compiler.git abc123)")
        plain = "clang (https://example.invalid/compiler.git abc123)"
        self.assertEqual(redact_url_credentials(plain), plain)

    def test_count_only_target_kernel_and_keep_resource_descriptor(self):
        source = """
        .amdgcn_target "amdgcn-amd-amdhsa--gfx936"
unrelated:
        v_exp_f32 v0, v1
.Lfunc_end0:
_Zkernel_flash_attn_v2_r64c32d64_kernel:
        v_mmac_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]
        ds_read_b64 v[4:5], v6
        s_waitcnt lgkmcnt(0)
        v_exp_f32 v0, v1
        // v_exp_f32 v0, v1
.LBB1_2:
        s_barrier
        s_cbranch_scc1 .LBB1_2
.Lfunc_end1:
        .amdhsa_kernel _Zkernel_flash_attn_v2_r64c32d64_kernel
        .amdhsa_group_segment_fixed_size 16384
        .amdhsa_private_segment_fixed_size 0
        .amdhsa_next_free_vgpr 72
        .amdhsa_next_free_sgpr 40
        .end_amdhsa_kernel
"""
        got = summarize_assembly(source)
        self.assertEqual(got["static_opcodes"], {
            "v_mmac_f32_16x16x16_f16": 1, "ds_read_b64": 1,
            "s_waitcnt": 1, "v_exp_f32": 1, "s_barrier": 1, "s_cbranch_scc1": 1})
        self.assertIn(".amdhsa_next_free_vgpr 72", got["resource_descriptor"])

    def test_missing_kernel_is_reported_not_counted_as_zero_instructions(self):
        got = summarize_assembly(".amdgcn_target gfx936\nother:\n v_exp_f32 v0, v1\n")
        self.assertIn("warning", got)
        self.assertNotIn("static_opcodes", got)

    def test_template_kernels_are_counted_separately(self):
        symbols = [f"_Z30flash_attn_v2_r64c32d64_kernelILb{b}EE" for b in (0, 1)]
        source = f"""
{symbols[0]}:
    v_cmp_lt_i32 vcc, v0, v1
    v_cndmask_b32 v2, v3, v4, vcc
    v_mmac_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]
.Lfunc_end0:
{symbols[1]}:
    v_mmac_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]
.Lfunc_end1:
"""
        infos = summarize_assemblies(source)
        self.assertEqual([x["symbol"] for x in infos], symbols)
        self.assertEqual(infos[0]["static_opcodes"]["v_cndmask_b32"], 1)
        self.assertNotIn("v_cndmask_b32", infos[1]["static_opcodes"])
        self.assertTrue(all(x["static_opcodes"]["v_mmac_f32_16x16x16_f16"] == 1 for x in infos))


if __name__ == "__main__":
    unittest.main()
