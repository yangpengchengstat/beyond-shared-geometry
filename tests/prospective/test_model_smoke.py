"""Smoke hardware checks use only construction rows and never become evidence."""
from copy import deepcopy

import numpy as np
import pytest

from bsg_next.smoke import run_model_smoke


class SmokeRuntime:
    def __init__(self, model, runtime_config, calls, change_zero=False):
        self.model_id = model["id"]
        self.model = None
        self.telemetry = []
        self.calls = calls
        self.closed = False
        self.change_zero = change_zero
        self.batch_size = runtime_config["batch_size"]

    def identity(self):
        return {"model_id":self.model_id, "revision":"local-mock", "batch_size":self.batch_size}

    def features(self, rows, layers, readout):
        assert all(row["split"] == "train" for row in rows)
        assert not any("test" in row["row_id"] or "pilot" in row["row_id"] or "validation" in row["row_id"] for row in rows)
        assert len(rows) == (8 if readout == "span_mean" else 1)
        assert readout in ("span_mean", "carrier_span")
        for row in rows:
            a,b = row["source_span" if readout == "span_mean" else "carrier_span"]
            assert row["text"][a:b] == (row["target"] if readout == "span_mean" else "Continue")
        self.calls.append((self.model_id, "features", len(rows), readout))
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start:start+self.batch_size]
            values = np.asarray([[row["state_id"], len(row["text"]), 1, 2] for row in batch],dtype="float32")
            yield list(range(start,start+len(batch))), {layer:values+layer for layer in layers}

    def score(self, prompt, candidates, patch=None, generation=False, max_new_tokens=None):
        assert generation and max_new_tokens == 2
        assert len(candidates) == 8  # All complete, multi-token D4 pose answers.
        self.calls.append((self.model_id, "score", patch is not None))
        if patch is not None:
            np.testing.assert_array_equal(patch.delta, np.zeros(4))
            assert prompt[slice(*patch.span)] == "Continue"
        values = np.arange(len(candidates),dtype=float)
        if patch is not None and self.change_zero:
            values[0] += .01
        return {"candidate_logprobs":values.tolist(), "candidate_token_counts":[4]*len(candidates),
                "generated_text":"two tokens"}

    def close(self):
        self.closed = True


def config():
    return {"seed":101, "technical_test":False,
            "models":[{"id":f"mock-{i}", "revision":"1"*40,"layers":[1,2],"hidden_dimension":4} for i in range(2)],
            "runtime":{"batch_size":3}}


def test_smoke_is_bounded_construction_only_sequential_and_checksum_resumable(tmp_path):
    calls, runtimes = [], []
    def factory(model, runtime_config):
        assert not runtimes or runtimes[-1].closed
        runtime = SmokeRuntime(model,runtime_config,calls)
        runtimes.append(runtime)
        return runtime
    cfg=config()
    first=run_model_smoke(cfg,tmp_path/"smoke",factory)
    assert first["status"] == "passed" and first["technical_test"] and not first["paper_eligible"]
    assert first["validation_pilot_test_outcomes_read"] is False
    assert first["operators_fitted"] is False and cfg["technical_test"] is False
    assert len(first["models"]) == 2 and all(runtime.closed for runtime in runtimes)
    assert len(calls) == 8  # Two feature calls and two scoring calls per model.
    def forbidden(*args):
        raise AssertionError("Verified smoke resume must not create or load a runtime")
    resumed=run_model_smoke(cfg,tmp_path/"smoke",forbidden)
    assert resumed["resumed"] and len(calls) == 8
    changed=deepcopy(cfg)
    changed["runtime"]["batch_size"] = 8
    with pytest.raises(ValueError,match="different configuration"):
        run_model_smoke(changed,tmp_path/"smoke",forbidden)
    cached=next((tmp_path/"smoke"/"caches").rglob("layer_1.npy"))
    array=np.load(cached,mmap_mode="r+")
    array[0,0]+=1
    array.flush()
    with pytest.raises(ValueError,match="checksum"):
        run_model_smoke(cfg,tmp_path/"smoke",forbidden)


def test_smoke_preserves_failed_zero_delta_and_closes_model(tmp_path):
    calls, runtimes = [], []
    def factory(model, runtime_config):
        runtime=SmokeRuntime(model,runtime_config,calls,change_zero=True)
        runtimes.append(runtime)
        return runtime
    with pytest.raises(RuntimeError,match="Zero-delta"):
        run_model_smoke(config(),tmp_path/"smoke",factory)
    assert runtimes[0].closed
    assert (tmp_path/"smoke"/"failure.json").exists()
    assert not (tmp_path/"smoke"/"completion.json").exists()
