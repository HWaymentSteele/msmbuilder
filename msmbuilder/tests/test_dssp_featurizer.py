import mdtraj as md
import numpy as np
from mdtraj.testing import eq

from msmbuilder.featurizer import DSSPFeaturizer
from msmbuilder.example_datasets import FsPeptide

t = FsPeptide().get().trajectories[0][:10]

def test_dssp_featurizer(t):

    residue_list = [x for x in t.topology.residues]
    if "ACE"==residue_list[0].name: #protein has caps
        n_residues = t.n_residues - 2
    else:
        n_residues = t.n_residues
        
    value = DSSPFeaturizer(simplified=True).partial_transform(t)
    assert value.shape == (t.n_frames, n_residues * 3), # simple DSSP should have 3 classes per residue

    value = DSSPFeaturizer(simplified=False).partial_transform(t)
    assert value.shape == (t.n_frames, n_residues * 8), # full DSSP should have 8 classes per residue
