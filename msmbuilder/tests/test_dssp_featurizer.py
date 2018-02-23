import mdtraj as md
import numpy as np
from mdtraj.testing import eq

from msmbuilder.featurizer import DSSPFeaturizer
from msmbuilder.example_datasets import FsPeptide

t = FsPeptide().get().trajectories[0][:10]

def test_dssp_featurizer():
    
    value = DSSPFeaturizer(simplified=True).partial_transform(t)
    assert value.shape == (t.n_frames, (t.n_residues - 2) * 3)

    value = DSSPFeaturizer(simplified=False).partial_transform(t)
    assert value.shape == (t.n_frames, (t.n_residues - 2) * 8)
