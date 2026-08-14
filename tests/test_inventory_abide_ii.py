from scripts.inventory_abide_ii import decide_gate, normalized_id, pcp_subject_ids


def test_normalized_id_removes_zero_padding():
    assert normalized_id("0050002") == "50002"
    assert normalized_id("29006.0") == "29006"


def test_pcp_subject_ids_extracts_only_aal_derivatives():
    keys = [
        "prefix/NYU_0050002_rois_aal.1D",
        "prefix/NYU_0050003_rois_cc200.1D",
        "prefix/BNI_29006_rois_aal.1D",
    ]
    assert pcp_subject_ids(keys) == {"50002", "29006"}


def test_gate_fails_closed_without_complete_exact_derivative():
    decision = decide_gate(
        exact_derivative_matches=0,
        main_participants=1114,
        direct_prefix_key_count=0,
        lle_site_count=21,
    )
    assert decision["decision"] == "FAIL"
    assert decision["model_evaluation_authorized"] is False
    assert len(decision["reasons"]) == 3


def test_gate_passes_only_for_complete_exact_derivative():
    decision = decide_gate(
        exact_derivative_matches=1114,
        main_participants=1114,
        direct_prefix_key_count=1114,
        lle_site_count=0,
    )
    assert decision["decision"] == "PASS"
    assert decision["model_evaluation_authorized"] is True
