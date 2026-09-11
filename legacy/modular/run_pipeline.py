#!/usr/bin/env python3
"""
Modularized entrypoint for the original NM pipeline.
This keeps the original logic and values, while moving helper/stage code into separate scripts.
"""
import os

import argparse
from pathlib import Path 

import nipype as ni
from nipype import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from nipype.interfaces.spm import Realign, Smooth, SPMCommand
from nipype.interfaces.ants import Registration, BrainExtraction, ApplyTransforms, N4BiasFieldCorrection
from nipype.algorithms.misc import Gunzip

from nilearn.image import smooth_img, mean_img
import nibabel as nib
import numpy as np

from templateflow.api import get

from phase1 import run_phase1
from segment import run_segment
from phase2 import run_phase2

# =================================================================
# Inputs
# =================================================================
safe_env = {'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '8', 'OMP_NUM_THREADS': '8'}

parser = argparse.ArgumentParser(description="Run NM_QC at the Subject Level")
parser.add_argument("--project_root", required=True, help="Root Data Folder (Location of Project Folder)", type=Path)
parser.add_argument("--output_root", required=True, help="Output Folder", type=Path)
parser.add_argument("--design_dir", required=True, help="Data Folder (Location of Subject Data)", type=Path)
parser.add_argument("--work_dir", required=True, help="Location of Work Directory", type=Path)
parser.add_argument("--n_procs", default=10, type=int)
parser.add_argument("--stage", choices=['1','segment', '2', 'all'], default='all',
                     help="Pipeline Stage to run. Defaults to 'all")
parser.add_argument("--manual_dir", default=None, type=Path, help="Directory of per-subject manual SN Segmentations (optional for similarity QC)")
args = parser.parse_args()

project_root = args.project_root
design_dir = args.design_dir
work_dir = args.work_dir
output_base = args.output_root
output_base.mkdir(parents=True, exist_ok=True)
manual_dir = args.manual_dir

subjects = sorted([p.name for p in design_dir.glob("sub-*")])
print(f"Found {len(subjects)} subjects: {subjects}")

# =================================================================
# Execution Flags
# =================================================================

run_phase1_flag = args.stage in ['1', 'all']
run_segment_flag = args.stage in ['segment', 'all']
run_phase2_flag = args.stage in ['2', 'all']

# =================================================================
# Templates and Masks
# =================================================================

t1_template = get('MNI152NLin2009cAsym', resolution=1, suffix='T1w', desc=None, extension='nii.gz')
if isinstance(t1_template, list):
    t1_template = t1_template[0]
else:
    t1_template = t1_template
print("Whole-Head T1 Candidates", t1_template)
t1_brain_template = get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='T1w', extension='nii.gz')
if isinstance(t1_brain_template, list):
    t1_brain_template=t1_brain_template[0]
print("Brain T1 Candidates", t1_brain_template)
brain_prob_template = get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='probseg', extension='nii.gz')
if isinstance(brain_prob_template, list):
    brain_prob_template = brain_prob_template[0]
print("Prob Brain Candidates", brain_prob_template)

combined_mask = f"{project_root}/Masks/MNI/MNI_Manual_Masks_combined.nii.gz"
SN_mask = f"{project_root}/Masks/MNI/MNI_SNc_Manual.nii.gz"
CP_mask = f"{project_root}/Masks/MNI/MNI_CP_Manual.nii.gz"
BrainStem_mask = f"{project_root}/Masks/MNI/MNI_Brainstem_Weight_Mask.nii.gz"

group_average_path = str(output_base / "Study-Specific_NM_Template.nii.gz")

if run_phase1_flag:
    run_phase1(
        design_dir=design_dir,
        work_dir=work_dir,
        output_base=output_base,
        subjects=subjects,
        t1_template=t1_template,
        t1_brain_template=t1_brain_template,
        brain_prob_template=brain_prob_template,
        BrainStem_mask=BrainStem_mask,
        group_average_path=group_average_path,
        n_procs=args.n_procs,
    )

if run_segment_flag:
    run_segment(
        project_root=project_root,
        output_base=output_base,
        t1_brain_template=t1_brain_template,
        combined_mask=combined_mask,
        group_average_path=group_average_path,
        CP_mask=CP_mask,
        SN_mask=SN_mask,
    )

if run_phase2_flag:
    run_phase2(
        output_base=output_base,
        work_dir=work_dir,
        subjects=subjects,
        group_average_path=group_average_path,
        SN_mask=SN_mask,
        CP_mask=CP_mask,
        manual_dir=manual_dir,
        n_procs=args.n_procs,
    )
