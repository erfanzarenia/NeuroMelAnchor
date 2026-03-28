import subprocess

import nibabel as nib
import numpy as np


def run_segment(project_root, output_base, t1_brain_template, combined_mask,
                group_average_path, CP_mask, SN_mask):
    t1_img = nib.load(str(t1_brain_template))
    placeholder = np.zeros(t1_img.shape, dtype=np.uint8)
    nib.save(nib.Nifti1Image(placeholder, t1_img.affine, t1_img.header), combined_mask)

    template_file = project_root / "Scripts" / "NEXUS_NM_Template_ITKsnap.xml"
    with open(template_file, 'r') as f:
        xml_data = f.read()

    xml_data = xml_data.replace("PROJECT_ROOT", str(project_root))
    xml_data = xml_data.replace("TARGET_T1_PATH", str(t1_brain_template))
    xml_data = xml_data.replace("TARGET_NM_PATH", group_average_path)
    xml_data = xml_data.replace("TARGET_MASK_PATH", combined_mask)

    workspace_path = f"{output_base}/Automated_Setup.itksnap"
    with open(workspace_path, 'w') as f:
        f.write(xml_data)

    print(f"{'='*60}", flush=True)
    print("Manaul Segmentation Required...", flush=True)
    print("1. Cerebral Penducle (CP) - Background reference Region", flush=True)
    print("2. Substantia Nigra (SN) - NM signal region", flush=True)
    print("1. Jump to Midbrain : Type  98  109   61 in the 'Cursor Position' box (center-left) and hit Enter.", flush=True)
    print("2. Center Cameras   : Click 'zoom to fit' below the image", flush=True)
    print("3. Zoom In          : Below 'Cursor Inspector', click Magnifcation Symbol and zoom into MidBrain .", flush=True)
    print(f"{'-'*60}", flush=True)
    print("Save your segmentation (Ctrl + S), then close the ITK-SNAP window to resume the pipeline.", flush=True)
    print(f"{'='*60}", flush=True)
    subprocess.run(['itksnap', '-w', workspace_path], check=True)

    combo_img = nib.load(combined_mask)
    combo_data = combo_img.get_fdata()
    if np.max(combo_data) == 0:
        raise ValueError("Segmentation Mask is Empty - re-run and save the labels.")

    nib.save(nib.Nifti1Image((combo_data == 1).astype(float), combo_img.affine, combo_img.header), CP_mask)
    nib.save(nib.Nifti1Image((combo_data == 2).astype(float), combo_img.affine, combo_img.header), SN_mask)

    CP_n = int((combo_data == 1).sum())
    SN_n = int((combo_data == 2).sum())
    print(f"{'='*60}")
    print(f"CP Voxels: {CP_n}  |  SN_voxels: {SN_n}")
    if CP_n < 50 or SN_n < 50:
        print("Warning: one or both masks have very few voxels. \n You may want to inspect segmentation before continuing.")
    print(f"{'='*60}")
