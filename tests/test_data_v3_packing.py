"""The streaming packer must be byte-identical to the original packer.

`pack_segments_streaming` exists only because `pack_segments` cannot fit the
pilot corpus in memory. It is therefore worth exactly nothing unless it produces
the same arrays — a silent difference here corrupts training data in a way no
downstream metric would attribute to the packer.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.data_v3 import EncodedSegment, pack_segments, pack_segments_streaming


def segment(ids, index=0, is_fim=False):
    return EncodedSegment(ids=list(ids), repository=f"owner/repo{index}",
                          path=f"src/file{index}.py", source="test",
                          is_fim=is_fim, fim_order="psm" if is_fim else None,
                          hole_kind="line" if is_fim else None)


def random_segments(rng, count, block_size):
    return [segment(rng.choices(range(3, 900), k=rng.randint(1, block_size + 1)),
                    index=i, is_fim=rng.random() < 0.4)
            for i in range(count)]


@pytest.mark.parametrize("block_size", [16, 64, 1024])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_streaming_matches_reference(block_size, seed):
    rng = random.Random(seed)
    segments = random_segments(rng, 60, block_size)
    want_inputs, want_labels, want_spans = pack_segments(segments, block_size, pad_id=1)
    got_inputs, got_labels, got_spans = pack_segments_streaming(
        iter(segments), block_size, pad_id=1, keep_spans=True, block_rows=7,
    )
    assert got_inputs.shape == want_inputs.shape
    assert got_inputs.dtype == want_inputs.dtype
    assert got_labels.dtype == want_labels.dtype
    assert np.array_equal(got_inputs, want_inputs)
    assert np.array_equal(got_labels, want_labels)
    assert got_spans == want_spans


def test_block_boundary_does_not_change_output():
    """Row-block size is an allocation detail and must not alter the result."""
    rng = random.Random(3)
    segments = random_segments(rng, 40, 32)
    reference = pack_segments_streaming(iter(segments), 32, pad_id=1, block_rows=1)[0]
    for block_rows in (2, 3, 5, 1000):
        other = pack_segments_streaming(iter(segments), 32, pad_id=1, block_rows=block_rows)[0]
        assert np.array_equal(reference, other), f"block_rows={block_rows} changed the packing"


def test_accepts_a_generator_without_materialising_it():
    """The whole point: the caller must be able to pass a generator."""
    consumed = []

    def source():
        for index in range(20):
            consumed.append(index)
            yield segment([5, 6, 7, 8], index=index)

    inputs, _, _ = pack_segments_streaming(source(), 16, pad_id=1, block_rows=4)
    assert len(consumed) == 20
    assert inputs.shape[1] == 16


def test_spans_are_dropped_unless_requested():
    rng = random.Random(11)
    segments = random_segments(rng, 12, 32)
    _, _, spans = pack_segments_streaming(iter(segments), 32, pad_id=1)
    assert spans == []


def test_empty_input_gives_empty_arrays():
    inputs, labels, spans = pack_segments_streaming(iter([]), 64, pad_id=1)
    assert inputs.shape == (0, 64)
    assert labels.shape == (0, 64)
    assert spans == []


def test_oversized_segment_is_rejected():
    with pytest.raises(ValueError):
        pack_segments_streaming(iter([segment(range(100))]), 16, pad_id=1)


def test_padding_is_masked_out_of_the_loss():
    """Positions after the real content must be -100 so they are not graded."""
    inputs, labels, _ = pack_segments_streaming(iter([segment([4, 5, 6])]), 8, pad_id=1)
    assert labels[0][0] == 5 and labels[0][1] == 6
    assert (labels[0][2:] == -100).all()
    assert (inputs[0][3:] == 1).all()
