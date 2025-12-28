import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from data import utils as du
from analysis import metrics
import seaborn as sns
import yaml
import shutil
import warnings
import concurrent.futures
import sys
import re
warnings.filterwarnings("ignore")
import time
from datetime import datetime
from openfold.np import residue_constants
import mdtraj as md
import argparse
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rc('font',family='Arial')
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    # "font.family": "serif",           # Use serif font
    # "font.serif": ["Times New Roman"], # Specify Times New Roman
    "axes.titlesize": 16,            # Title font size
    "axes.labelsize": 14,            # Axis labels font size
    "xtick.labelsize": 12,           # X-axis tick font size
    "ytick.labelsize": 12,           # Y-axis tick font size
    "legend.fontsize": 12,           # Legend font size
    "legend.title_fontsize": 12,      # Legend title font size
    # "text.usetex": True
})

def file_generate(inference_dir, type=None):

    def natural_sort_key(name):
        return int(''.join(filter(str.isdigit, name)) or 0)
    
    All_Results = pd.DataFrame()
    SinglePDB_Metrics = pd.DataFrame()

    # get and sort length_folder
    for length_folder in sorted(os.listdir(inference_dir), key=natural_sort_key):
        length_path = os.path.join(inference_dir, length_folder)
        if os.path.isdir(length_path):
            length = int(length_folder.split('_')[-1])

            # get and sort sample_folder
            for sample_folder in sorted(os.listdir(length_path), key=natural_sort_key):
                sample_path = os.path.join(length_path, sample_folder)
                if os.path.isdir(sample_path):
                    Model_sample_index = int(sample_folder.split('_')[-1])
                    if type.lower() in ['frameflow', 'FrameFlow','qflow']:
                        Model_sample_pdb_path = os.path.join(sample_path, 'sample.pdb')
                    else:
                        Model_sample_pdb_path = os.path.join(sample_path, 'sample_1.pdb')
                    

                    SinglePDB_Metrics_row = {'length': length, 'pdb_path': Model_sample_pdb_path}
                    self_consistency_path = os.path.join(sample_path, 'self_consistency')
                    if os.path.isdir(self_consistency_path):
                        csv_path = os.path.join(self_consistency_path, 'sc_results.csv')
                        
                        if os.path.isfile(csv_path):
                            sc_results = pd.read_csv(csv_path, skiprows=[1])
                            sc_results = sc_results.rename(columns={'Unnamed: 0': 'ESMF_index', 'sample_path': 'ESMF_sample_path'})

                            sc_results.index = range(1, len(sc_results) + 1)
                            sc_results.insert(0, 'length', length)  
                            sc_results.insert(1, 'Model_sample_index', Model_sample_index)  
                            sc_results.insert(2, 'Model_sample_pdb_path', Model_sample_pdb_path)  

                            if sc_results['rmsd'].min() < 2 :
                                sc_results.insert(3, 'Designable', "True")
                            else:
                                sc_results.insert(3, 'Designable', "False")
                            sc_results.insert(5, 'min_rmsd', sc_results['rmsd'].min())
                            sc_results.insert(6, 'max_tm_score', sc_results['tm_score'].max())
                            SinglePDB_Metrics_row.update({
                                'min_rmsd': sc_results['rmsd'].min(),
                                'max_tm_score': sc_results['tm_score'].max()
                            })
                            # put some not frequently used columns to the end
                            esmf_col = sc_results.pop('ESMF_sample_path')
                            sc_results['ESMF_sample_path'] = esmf_col
                            header_col = sc_results.pop('header')
                            sc_results['header'] = header_col
                            sequence_col = sc_results.pop('sequence')
                            sc_results['sequence'] = sequence_col
                            All_Results = pd.concat([All_Results, sc_results], ignore_index=True)
                    
                    
                    metrics = eval_secondary_structure(Model_sample_pdb_path)
                    SinglePDB_Metrics_row.update({
                                'helix_percent': metrics['helix_percent'],
                                'strand_percent': metrics['strand_percent'],
                                'coil_percent': metrics['coil_percent'],
                                'non_coil_percent': metrics['non_coil_percent'],
                                'radius_of_gyration': metrics['radius_of_gyration'],
                                'ca_ca_deviation': metrics['ca_ca_deviation'],
                                'ca_ca_valid_percent': metrics['ca_ca_valid_percent'],
                                'num_ca_ca_clashes': metrics['num_ca_ca_clashes']
                            })
                    SinglePDB_Metrics = SinglePDB_Metrics._append(SinglePDB_Metrics_row, ignore_index=True)

    output_dir_1 = os.path.join(inference_dir, 'All_Results_Origin.csv')
    All_Results.to_csv(output_dir_1, index=False)

    output_dir_2 = os.path.join(inference_dir, 'All_Sampled_PDB_Designable.txt')
    All_Results[All_Results['Designable'] == "True"]['Model_sample_pdb_path'].drop_duplicates().to_csv(output_dir_2, index=False, header=False)

    output_dir_3 = os.path.join(inference_dir, 'All_Sampled_PDB.txt')
    All_Results['Model_sample_pdb_path'].drop_duplicates().to_csv(output_dir_3, index=False, header=False)

    output_dir_4 = os.path.join(inference_dir, 'All_Sampled_PDB_and_Length_Designable.csv')
    All_Results[All_Results['Designable'] == "True"][['length','Model_sample_pdb_path']].drop_duplicates().to_csv(output_dir_4, index=False)

    output_dir_5 = os.path.join(inference_dir, 'All_Sampled_PDB_and_Length.csv')
    All_Results[['length','Model_sample_pdb_path']].drop_duplicates().to_csv(output_dir_5, index=False)

    output_dir_6 = os.path.join(inference_dir, 'Single_PDB_Metrics.csv')
    SinglePDB_Metrics.to_csv(output_dir_6, index=False)


    print("############### File Generation End ###############")


def plot_time(inference_dir, type):

    file_candidates = ["inference_conf.yaml", "config.yaml"]
    selected_file = None
    for file_name in file_candidates:
        if os.path.exists(os.path.join(inference_dir, file_name)):
            selected_file = file_name
            break
    yaml_dir = os.path.join(inference_dir, selected_file)
    with open (yaml_dir, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    inference_scaling = None
    model_type = None
    dataset = None
    rectify = None
    if type.lower() in ['foldflow']:
        num_t = conf['inference']['flow']['num_t']
        noise_scale = conf['inference']['flow']['noise_scale']
        sample_per_length = conf['inference']['samples']['samples_per_length']
        seq_per_sample = conf['inference']['samples']['seq_per_sample']
        inference_scaling = conf['flow_matcher']['so3']['inference_scaling']
        model_type = 'FoldFlow_Base'
        if conf['flow_matcher']['ot_plan'] == True:
            if conf['flow_matcher']['stochastic_paths'] == True:
                model_type = 'FoldFlow_SFM'
            else:
                model_type = 'FoldFlow_OT'

    elif type.lower() in ['framediff']:
        num_t = conf['inference']['diffusion']['num_t']
        noise_scale = conf['inference']['diffusion']['noise_scale']
        sample_per_length = conf['inference']['samples']['samples_per_length']
        seq_per_sample = conf['inference']['samples']['seq_per_sample']

    elif type.lower() in ['frameflow']:
        num_t = conf['inference']['interpolant']['sampling']['num_timesteps']
        noise_scale = None
        sample_per_length = conf['inference']['samples']['samples_per_length']
        seq_per_sample = conf['inference']['samples']['seq_per_sample']

    elif type.lower() in ['qflow']:
        num_t = conf['inference']['interpolant']['sampling']['num_timesteps']
        noise_scale = None
        sample_per_length = conf['inference']['samples']['samples_per_length']
        seq_per_sample = conf['inference']['samples']['seq_per_sample']
        dataset = conf['data']['dataset']
        rectify = conf['data'].get('rectify', False)

    elif type.lower() in ['rfdiffusion']:
        num_t = conf['diffuser']['T']
        noise_scale = conf['denoiser']['noise_scale_ca']
        sample_per_length = conf['inference']['num_designs']
        seq_per_sample = conf['inference']['seq_per_sample']

    elif type.lower() in ['genie2']:
        num_t = conf['inference_step']
        noise_scale = conf['scale']
        sample_per_length = conf['num_samples']
        seq_per_sample = conf['seq_per_sample']
    else:
        raise ValueError(f"Unknown type: {type}")
        

    with open(os.path.join(inference_dir, "Metrics.txt"), "w") as f:
        f.write("\n############# Evaluation Results #############\n")
        f.write(f"Model: {type}\n")
        if model_type:
            f.write(f"Model Type: {model_type}\n")
        f.write(f"num_t: {num_t}\n")
        f.write(f"noise_scale: {noise_scale}\n")
        f.write(f"sample_per_length: {sample_per_length}\n")
        f.write(f"seq_per_sample: {seq_per_sample}\n")
        if inference_scaling:
            f.write(f"SO3_inference_scaling: {inference_scaling}\n")
        if dataset:
            f.write(f"Dataset: {dataset}\n")
        if rectify is not None:
            f.write(f"Rectify: {rectify}\n")

    if os.path.exists(os.path.join(inference_dir, 'Single_PDB_Metrics.csv')):
        SS_Metrics = pd.read_csv(os.path.join(inference_dir, 'Single_PDB_Metrics.csv'))
        sns.set(style='white')
        g = sns.jointplot(data=SS_Metrics, x='helix_percent', y='strand_percent', kind='scatter', color='#4CB391')
        g.set_axis_labels('Helix Percent', 'Strand Percent')
        g.fig.suptitle(f'{type} Secondary Structure Percentages')
        g.savefig(os.path.join(inference_dir, f'Secondary_Structure_Percentages_{type}.svg'))

    time_record_dir = os.path.join(inference_dir, 'time_records.csv')
    if not os.path.isfile(time_record_dir):
        print("Time record file not found!")
        return
    time_record = pd.read_csv(time_record_dir)
    avg_times = time_record.groupby('length')[['sample_time', 'eval_time', 'total_time']].mean().reset_index()
    sns.set(font="Arial")
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 8), dpi=200)
    sns.lineplot(x='length', y='sample_time', data=avg_times, marker='o', label='Sample Time')
    sns.lineplot(x='length', y='eval_time', data=avg_times, marker='o', label='Eval Time')
    sns.lineplot(x='length', y='total_time', data=avg_times, marker='o', label='Total Time')
    plt.xlabel('Length', fontsize=14)
    plt.ylabel('Average Time (seconds)', fontsize=14)
    plt.title(f'{type} Average Times per Sample, num_t={num_t}, num_seq={seq_per_sample}')
    plt.legend(title='Time Type')
    plt.savefig(os.path.join(inference_dir, f'Average_Times_per_Sample_{type}_num_t_{num_t}_num_seq_{seq_per_sample}.png'))



def designability_calculate(inference_dir):

    All_Results = pd.read_csv(os.path.join(inference_dir, 'All_Results_Origin.csv'))

    tm_score_mean = All_Results['tm_score'].mean()
    tm_score_std = All_Results['tm_score'].std()
    tm_score_greater_0_5_ratio = (All_Results['tm_score'] > 0.5).mean()

    max_tm_score_mean = All_Results['max_tm_score'].mean()
    max_tm_score_std = All_Results['max_tm_score'].std()
    max_tm_score_greater_0_5_ratio = (All_Results['max_tm_score'] > 0.5).mean()

    rmsd_mean = All_Results['rmsd'].mean()
    rmsd_std = All_Results['rmsd'].std()
    rmsd_below_2_ratio = (All_Results['rmsd'] < 2).mean()

    min_rmsd_mean = All_Results['min_rmsd'].mean()
    min_rmsd_std = All_Results['min_rmsd'].std()
    min_rmsd_below_2_ratio = (All_Results['min_rmsd'] < 2).mean()
    
    
    max_tm_score_per_length = []
    min_rmsd_per_length = []
    grouped = All_Results.groupby('length')
    for length, group in grouped:
        max_tm_score_per_length.append((group['max_tm_score'] > 0.5).mean())
        min_rmsd_per_length.append((group['min_rmsd'] < 2).mean())

    grouped_tm_score_mean = np.mean(max_tm_score_per_length)
    grouped_tm_score_std = np.std(max_tm_score_per_length)
    grouped_rmsd_mean = np.mean(min_rmsd_per_length)
    grouped_rmsd_std = np.std(min_rmsd_per_length)


    print("\n########### Designability Calculation End ###########")
    print(f"grouped_rmsd_below_2_ratio: {grouped_rmsd_mean:.3f} ± {grouped_rmsd_std:.3f}\n\n")
    print(f"min_rmsd_range: {min_rmsd_mean:.3f} ± {min_rmsd_std:.3f}\n")
    print(f"max_tm_score_range: {max_tm_score_mean:.3f} ± {max_tm_score_std:.3f}\n")
    print(f"grouped_tm_score_greater_0.5_ratio: {grouped_tm_score_mean:.3f} ± {grouped_tm_score_std:.3f}\n\n")

    print(f"tm_score_mean: {tm_score_mean}")
    print(f"tm_score_std: {tm_score_std}")
    print(f"tm_score_range: {tm_score_mean:.3f} ± {tm_score_std:.3f}")
    print(f"tm_score_greater_0_5_ratio: {tm_score_greater_0_5_ratio}\n")

    print(f"max_tm_score_mean: {max_tm_score_mean}")
    print(f"max_tm_score_std: {max_tm_score_std}")
    print(f"max_tm_score_range: {max_tm_score_mean:.3f} ± {max_tm_score_std:.3f}")
    print(f"max_tm_score_greater_0_5_ratio: {max_tm_score_greater_0_5_ratio}\n")
    print(f"grouped_tm_score_greater_0.5_ratio: {grouped_tm_score_mean:.3f} ± {grouped_tm_score_std:.3f}\n")

    print(f"rmsd_mean: {rmsd_mean}")
    print(f"rmsd_std: {rmsd_std}")
    print(f"rmsd_range: {rmsd_mean:.3f} ± {rmsd_std:.3f}")
    print(f"rmsd_below_2_ratio: {rmsd_below_2_ratio}\n")

    print(f"min_rmsd_mean: {min_rmsd_mean}")
    print(f"min_rmsd_std: {min_rmsd_std}")
    print(f"min_rmsd_range: {min_rmsd_mean:.3f} ± {min_rmsd_std:.3f}")
    print(f"min_rmsd_below_2_ratio: {min_rmsd_below_2_ratio}")
    print(f"grouped_rmsd_below_2_ratio: {grouped_rmsd_mean:.3f} ± {grouped_rmsd_std:.3f}")
    print("\n")

    with open(os.path.join(inference_dir, "Metrics.txt"), "a") as f:
        f.write("\n########### Designability Calculation ###########\n")
        f.write(f"grouped_rmsd_below_2_ratio: {grouped_rmsd_mean:.3f} ± {grouped_rmsd_std:.3f}\n")
        f.write(f"min_rmsd_range: {min_rmsd_mean:.3f} ± {min_rmsd_std:.3f}\n")
        f.write(f"max_tm_score_range: {max_tm_score_mean:.3f} ± {max_tm_score_std:.3f}\n")
        f.write(f"grouped_tm_score_greater_0.5_ratio: {grouped_tm_score_mean:.3f} ± {grouped_tm_score_std:.3f}\n\n")

        f.write(f"tm_score_mean: {tm_score_mean}\n")
        f.write(f"tm_score_std: {tm_score_std}\n")
        f.write(f"tm_score_range: {tm_score_mean:.3f} ± {tm_score_std:.3f}\n")
        f.write(f"tm_score_greater_0_5_ratio: {tm_score_greater_0_5_ratio}\n\n")
        
        f.write(f"max_tm_score_mean: {max_tm_score_mean}\n")
        f.write(f"max_tm_score_std: {max_tm_score_std}\n")
        f.write(f"max_tm_score_range: {max_tm_score_mean:.3f} ± {max_tm_score_std:.3f}\n")
        f.write(f"max_tm_score_greater_0_5_ratio: {max_tm_score_greater_0_5_ratio}\n\n")
        f.write(f"grouped_tm_score_greater_0.5_ratio: {grouped_tm_score_mean:.3f} ± {grouped_tm_score_std:.3f}\n\n")
        
        f.write(f"rmsd_mean: {rmsd_mean}\n")
        f.write(f"rmsd_std: {rmsd_std}\n")
        f.write(f"rmsd_range: {rmsd_mean:.3f} ± {rmsd_std:.3f}\n")
        f.write(f"rmsd_below_2_ratio: {rmsd_below_2_ratio}\n\n")
        
        f.write(f"min_rmsd_mean: {min_rmsd_mean}\n")
        f.write(f"min_rmsd_std: {min_rmsd_std}\n")
        f.write(f"min_rmsd_range: {min_rmsd_mean:.3f} ± {min_rmsd_std:.3f}\n")
        f.write(f"min_rmsd_below_2_ratio: {min_rmsd_below_2_ratio}\n")
        f.write(f"grouped_rmsd_below_2_ratio: {grouped_rmsd_mean:.3f} ± {grouped_rmsd_std:.3f}\n\n")

def calc_tm_score_wrapper(feats_pair):
    feats_1, feats_2 = feats_pair
    _, tm_score = metrics.calc_tm_score(
        feats_1["bb_positions"],
        feats_2["bb_positions"],
        du.aatype_to_seq(feats_2["aatype"]),
        du.aatype_to_seq(feats_2["aatype"])
    )
    return tm_score

def diversity_calculate(inference_dir, type):
    '''
    type = 'Designable' or 'All'
    '''
    if type.lower() in ['designable']:
        All_Sampled_PDB_Path = os.path.join(inference_dir, 'All_Sampled_PDB_and_Length_Designable.csv')
    elif type.lower() in ['all']:
        All_Sampled_PDB_Path = os.path.join(inference_dir, 'All_Sampled_PDB_and_Length.csv')
    All_Sampled_PDB = pd.read_csv(All_Sampled_PDB_Path)
    diversity_per_length = []
    grouped = All_Sampled_PDB.groupby('length')
    for length, group in grouped:
        pdb_paths = group['Model_sample_pdb_path'].tolist()
        length_samples_feats = []
        for pdb_path in pdb_paths:
            sample_feats = du.parse_pdb_feats("sample", pdb_path)
            length_samples_feats.append(sample_feats)

        if len(length_samples_feats) > 1:
            pairwise_tm_scores = []

            for feats_1, feats_2 in itertools.combinations(length_samples_feats, 2):
                _, tm_score = metrics.calc_tm_score(
                    feats_1["bb_positions"],
                    feats_2["bb_positions"],
                    du.aatype_to_seq(feats_2["aatype"]),
                    du.aatype_to_seq(feats_2["aatype"])
                )
                pairwise_tm_scores.append(tm_score)

            if pairwise_tm_scores:
                length_diversity = sum(pairwise_tm_scores) / len(pairwise_tm_scores)
                diversity_per_length.append(length_diversity)

    total_diversity = sum(diversity_per_length) / len(diversity_per_length)
    print(f"\n########### Diversity Calculation (Pairwise-TM, {type}) ###########")
    print(f"total_diversity: {total_diversity:.3f}")
    with open(os.path.join(inference_dir, "Metrics.txt"), "a") as f:
        f.write(f"\n########### Diversity Calculation (Pairwise-TM, {type}) ###########\n")
        f.write(f"total_diversity: {total_diversity:.3f}\n")




def run_foldseek(inference_dir, script_path, output_dir, database = "pdb", result_summary = None, dataset_dir = None):
    '''
    inference_dir: the directory of the inference results
    script_path: the path of the script to run FoldSeek. A .sh file
    output_dir: the output directory of the FoldSeek results
    database: the database used in FoldSeek
    result_summary: the summary file of the FoldSeek results
    dataset_dir: the directory of the foldseek dataset. Cannot be None !!
    '''
    pdb_list_dir = os.path.join(inference_dir, 'All_Sampled_PDB.txt')
    pdb_list_dir_designable = os.path.join(inference_dir, 'All_Sampled_PDB_Designable.txt')
    if result_summary == None:
        result_summary = os.path.join(output_dir, 'summary_tmscore.csv')
    try: 
        subprocess.run(
            [script_path, pdb_list_dir, pdb_list_dir_designable, output_dir, database, result_summary],
            check = True,
            cwd = dataset_dir
        )
    except subprocess.CalledProcessError as e:
        print(f"Script execution failed with error: {e}")
    except FileNotFoundError:
        print("[Foldseek] Script file not found! Please check the path.")

def foldseek_calculate(foldseek_output_dir):
    df = pd.read_csv(os.path.join(foldseek_output_dir, 'summary_tmscore.csv'))
    df_designable = df[df['Designable'] == 1]
    print("\n########### Novelty Calculation (All) ###########")
    print('tmscore mean',df['Max TM-score'].mean())
    print('tmscore std',df['Max TM-score'].std())
    print('tmscore range', f"{df['Max TM-score'].mean():.3f} ± {df['Max TM-score'].std():.3f}")
    print('tmscore < 0.5 ratio', (df['Max TM-score'] < 0.5).mean())
    print("\n########### Novelty Calculation (Designable) ###########")
    print('tmscore mean',df_designable['Max TM-score'].mean())
    print('tmscore std',df_designable['Max TM-score'].std())
    print('tmscore range', f"{df_designable['Max TM-score'].mean():.3f} ± {df_designable['Max TM-score'].std():.3f}")
    print('tmscore < 0.5 ratio', (df_designable['Max TM-score'] < 0.5).mean())

    with open(os.path.join(foldseek_output_dir, "Metrics.txt"), "a") as f:
        f.write("\n########### Novelty Calculation (Designable) ###########\n")
        f.write(f'tmscore mean: {df_designable["Max TM-score"].mean()}\n')
        f.write(f'tmscore std: {df_designable["Max TM-score"].std()}\n')
        f.write(f"tmscore range: {df_designable['Max TM-score'].mean():.3f} ± {df_designable['Max TM-score'].std():.3f}\n")
        f.write(f'tmscore < 0.5 ratio: {(df_designable["Max TM-score"] < 0.5).mean()}\n')
        
        f.write("\n########### Novelty Calculation (All) ###########\n")
        f.write(f'tmscore mean: {df["Max TM-score"].mean()}\n')
        f.write(f'tmscore std: {df["Max TM-score"].std()}\n')
        f.write(f"tmscore range: {df['Max TM-score'].mean():.3f} ± {df['Max TM-score'].std():.3f}\n")
        f.write(f'tmscore < 0.5 ratio: {(df["Max TM-score"] < 0.5).mean()}\n')

        
def calc_mdtraj_metrics(pdb_path):
    """
    use mdtraj to calculate secondary structure and radius of gyration
    """
    traj = md.load(pdb_path)
    pdb_ss = md.compute_dssp(traj, simplified=True)
    pdb_coil_percent = np.mean(pdb_ss == "C")
    pdb_helix_percent = np.mean(pdb_ss == "H")
    pdb_strand_percent = np.mean(pdb_ss == "E")
    pdb_ss_percent = pdb_helix_percent + pdb_strand_percent
    pdb_rg = md.compute_rg(traj)[0]
    return {
        "non_coil_percent": pdb_ss_percent,
        "coil_percent": pdb_coil_percent,
        "helix_percent": pdb_helix_percent,
        "strand_percent": pdb_strand_percent,
        "radius_of_gyration": pdb_rg,
    }


def calc_ca_ca_metrics(ca_pos, bond_tol=0.1, clash_tol=1.0):

    ca_pos = ca_pos * 10 
    ca_bond_dists = np.linalg.norm(
        ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1
    )[1:] 
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - residue_constants.ca_ca))
    ca_ca_valid = np.mean(ca_bond_dists < (residue_constants.ca_ca + bond_tol))
    
    ca_ca_dists2d = np.linalg.norm(
        ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1
    )
  
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=1) > 0)]
    clashes = inter_dists < clash_tol
    return {
        'ca_ca_deviation': ca_ca_dev,
        'ca_ca_valid_percent': ca_ca_valid,
        'num_ca_ca_clashes': np.sum(clashes),
    }


def eval_secondary_structure(pdb_path):

    mdtraj_metrics = calc_mdtraj_metrics(pdb_path)

    traj = md.load(pdb_path)
    ca_indices = traj.topology.select('name CA')
    if len(ca_indices) == 0:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    ca_pos = traj.xyz[0, ca_indices, :]  

    ca_ca_metrics = calc_ca_ca_metrics(ca_pos)

    combined_metrics = {**mdtraj_metrics, **ca_ca_metrics}
    return combined_metrics


def calc_additional_metrics(inference_dir, model_type, base_dir=None, time_folder=None, length_values=None):
    """
    Calculate additional secondary structure and CA-CA related metrics and write to Metrics.txt.
    para:
    inference_dir: the directory of the inference results
    model_type: the type of the model
    base_dir: the base directory of the inference results
    time_folder: the time folder of the inference results
    length_values: the length values of the inference results
    """

    # if base_dir and time_folder are not provided, try to infer them from inference_dir
    if base_dir is None or time_folder is None:
        # assume inference_dir is in the format of base_dir/time_folder
        # e.g. ：/data/.../run_2025-01-15_10-48-15
        base_dir = os.path.dirname(inference_dir)
        time_folder = os.path.basename(inference_dir)
    
    if length_values is None:
            length_values = []
            for item in os.listdir(inference_dir):
                item_path = os.path.join(inference_dir, item)
                if os.path.isdir(item_path) and item.startswith("length_"):
                    try:
                        length = int(item.split('_')[-1])
                        length_values.append(length)
                    except ValueError:
                        logging.warning(f"Cannot extract length value from folder {item}, skipping.")
            length_values = sorted(length_values)
            if not length_values:
                logging.error("No length values found in inference_dir")
                return
            logging.info(f"Dynamically obtained length_values: {length_values}")
    
    time_path = os.path.join(base_dir, time_folder)
    
    all_metrics = {
        'helix_percent': [],
        'strand_percent': [],
        'coil_percent': [],
        'radius_of_gyration': [],
        'ca_ca_deviation': [],
        'ca_ca_valid_percent': [],
        'num_ca_ca_clashes': []
    }
    
    for length in length_values:
        length_folder = f'length_{length}'
        length_path = os.path.join(time_path, length_folder)
        helix = []
        strand = []
        coil = []
        rg = []
        ca_ca_deviation = []
        ca_ca_valid_percent = []
        num_ca_ca_clashes = []
        
        if os.path.exists(length_path):
            num_samples =  len(os.listdir(length_path))
            for sample_num in range(num_samples):
                sample_folder = f'sample_{sample_num}'
                sample_path = os.path.join(length_path, sample_folder)
                
                if os.path.exists(sample_path):
                    pdb_file_1 = os.path.join(sample_path, 'sample.pdb')
                    pdb_file_2 = os.path.join(sample_path, 'sample_1.pdb')
                    if os.path.exists(pdb_file_1):
                        pdb_file = pdb_file_1
                    elif os.path.exists(pdb_file_2):
                        pdb_file = pdb_file_2
                    else:
                        logging.warning(f'File not found: {pdb_file_1}')
                        continue

                    if os.path.exists(pdb_file):
                        try:
                            metrics = eval_secondary_structure(pdb_file)
                            helix.append(metrics['helix_percent'])
                            strand.append(metrics['strand_percent'])
                            coil.append(metrics['coil_percent'])
                            rg.append(metrics['radius_of_gyration'])
                            ca_ca_deviation.append(metrics['ca_ca_deviation'])
                            ca_ca_valid_percent.append(metrics['ca_ca_valid_percent'])
                            num_ca_ca_clashes.append(metrics['num_ca_ca_clashes'])
                        except Exception as e:
                            logging.error(f'Error processing {pdb_file}: {e}')
                    else:
                        logging.warning(f'File not found: {pdb_file}')
                else:
                    logging.warning(f'Sample folder not found: {sample_path}')
        else:
            logging.warning(f'Length folder not found: {length_path}')
        
        if helix:
            all_metrics['helix_percent'].extend(helix)
            all_metrics['strand_percent'].extend(strand)
            all_metrics['coil_percent'].extend(coil)
            all_metrics['radius_of_gyration'].extend(rg)
            all_metrics['ca_ca_deviation'].extend(ca_ca_deviation)
            all_metrics['ca_ca_valid_percent'].extend(ca_ca_valid_percent)
            all_metrics['num_ca_ca_clashes'].extend(num_ca_ca_clashes)
        else:
            logging.info(f'No valid samples found in {length_path}')
    
    metrics_df = pd.DataFrame(all_metrics)
    
    if not metrics_df.empty:
        metrics_summary = metrics_df.agg(['mean', 'std'])

        logging.info(f"\n########### Additional Metrics Summary for {model_type} ###########")
        logging.info(metrics_summary)

        with open(os.path.join(inference_dir, "Metrics.txt"), "a") as f:
            f.write(f"\n########### Additional Metrics Summary for {model_type} ###########\n")
            f.write(metrics_summary.to_string())
            f.write("\n")
    else:
        logging.warning("No additional metrics were calculated.")

def clean_folder(inference_dir):
    if not os.path.exists(inference_dir):
        print(f"Folder {inference_dir} do not exist!")
        return

    for item_name in os.listdir(inference_dir):
        item_path = os.path.join(inference_dir, item_name)
        if os.path.isfile(item_path):
            if item_name not in {"inference_conf.yaml", "time_records.csv", "config.yaml", "config.yml", "inferece_conf.yml","prot_df.csv"}:
                os.remove(item_path)
                print(f"Deleted file: {item_path}")

        elif os.path.isdir(item_path):
            if not item_name.startswith("length"):
                shutil.rmtree(item_path)
                print(f"Deleted folder: {item_path}")


def plot_helix_strand(inference_dir):
    df = pd.read_csv("analysis/data/ReQFlow_PDB_Helix.csv")
    df_new = pd.read_csv(os.path.join(inference_dir, "Single_PDB_Metrics.csv"))
    df_new = df_new[['helix_percent', 'strand_percent']]
    df_new.insert(0, "method", "current")

    df = pd.concat([df, df_new])


    df =  df[~df.select_dtypes(include=['number']).apply(lambda row: (row > 1).any() or (row < 0).any(), axis=1)]
    # 2. 获取所有的 method
    methods = df['method'].unique()

    # 3. 先遍历所有 method，计算 2D 直方图（bins=10），
    #    并做归一化(除以总和 -> frequency)
    histograms = []
    xedges_list = []
    yedges_list = []

    for m in methods:
        sub_df = df[df['method'] == m]
        H, xedges, yedges = np.histogram2d(
            sub_df['helix_percent'],
            sub_df['strand_percent'],
            bins=10,
            range=[[0, 1], [0, 1]],
        )
        H = H / H.sum()  # 归一化
        H = np.clip(H, 0, 0.15)  
        histograms.append(H)
        xedges_list.append(xedges)
        yedges_list.append(yedges)

    # 4. 找到所有热力图中的最大值，用于设定统一的 vmax
    max_val = max(H.max() for H in histograms)

    # 5. 自定义一个新的 colormap，使最低值(0)对应白色
    #    先从 Greens 取出 256 个颜色，然后将第 0 个颜色设置为白色
    base_cmap = plt.cm.get_cmap('Reds', 256)       # 从 'Greens' 获取 256 级颜色
    newcolors = base_cmap(np.linspace(0, 0.7, 256))    # 转成 RGBA 数组
    newcolors[0, :] = np.array([1, 1, 1, 1])         # 第一个颜色改为白色(RGBA=1,1,1,1)
    white_cmap = mcolors.ListedColormap(newcolors)   # 生成新的 cmap

    # 6. 创建绘图区域：1 行 5 列
    #    figsize 你可以自行调整，下面示例用 (20,4)；可以适当减小/增大
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharex=True, sharey=True)

    # 7. 调整子图与画布边缘的间距；这里留出右侧空间给 colorbar
    #    wspace 减小以压缩子图之间的水平间距
    plt.subplots_adjust(left=0.05, right=0.88, bottom=0.15, top=0.85, wspace=0.2)

    # 8. 逐个子图绘制
    for i, m in enumerate(methods):
        ax = axes[i]
        H = histograms[i]
        # xedges = xedges_list[i]  # 如果想在坐标轴上精确还原 x,y 范围，可以修改 extent
        # yedges = yedges_list[i]
        
        # extent=[0,1,0,1] 表示将 x,y 范围都固定到 [0,1]
        # 如果你想保留原始分箱范围，可以用: 
        #    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(
            H.T,
            origin='lower',
            aspect='equal',
            extent=[0,1,0,1],
            vmin=0,
            vmax=max_val,
            cmap=white_cmap  # 使用自定义的 cmap
        )
        
        ax.set_title(m, fontsize=20)
        ax.set_xlabel("Helix percent", fontsize=20)
        if i == 0:
            ax.set_ylabel("Strand percent", fontsize=20)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis='both', labelsize=18)


    # 9. 在右侧手动添加 colorbar
    #    先在 figure 上开辟一个坐标轴区域 [x0, y0, width, height]
    #    这里可以根据实际需求微调
    cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Frequency", fontsize=18)
    ticks = [0.0, 0.05, 0.1, 0.15]
    cbar.set_ticks(ticks)
    tick_labels = [f"{val:.2f}" for val in ticks]
    tick_labels[-1] = ">0.15"  # 最后一个改成 ">0.4"
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=18)
    plt.savefig(os.path.join(inference_dir, "combined_helix_strand_percent_heatmap.pdf"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run QFlow evaluation.")
    # Required arguments
    parser.add_argument("--inference_dir", required=True, help="Directory containing inference results.")
    parser.add_argument("--script_path", default="/home/zichong_wang/ReQFlow/analysis/run_foldseek_parallel.sh", help="Absolute Path to the FoldSeek script.")
    parser.add_argument("--dataset_dir", default="/home/zichong_wang/ReQFlow/foldseek/foldseek_db", help="Directory containing the FoldSeek dataset.")

    # Optional arguments with defaults
    parser.add_argument("--database", default="pdb", help="Database to use (e.g., pdb).")
    parser.add_argument("--type", default="qflow", help="Type of evaluation (qflow, FrameFlow, FoldFlow, FrameDiff, Genie2, RFdiffusion).")


    args = parser.parse_args()

    inference_dir = args.inference_dir
    script_path = args.script_path
    dataset_dir = args.dataset_dir
    output_dir = inference_dir  # Use inference_dir if output_dir is not provided
    database = args.database
    type = args.type

    start_time = time.time()
    clean_folder(inference_dir)
    file_generate(inference_dir, type=type)
    plot_time(inference_dir, type=type)
    designability_calculate(inference_dir)
    calc_additional_metrics(inference_dir, type, base_dir=None, time_folder=None, length_values=None)
    plot_helix_strand(inference_dir)

    # run_foldseek(inference_dir, script_path, output_dir, database, dataset_dir=dataset_dir)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     future_diversity_designable = executor.submit(diversity_calculate, inference_dir, type='Designable')
    #     future_diversity_all = executor.submit(diversity_calculate, inference_dir, type='All')

    # # wait for all the futures to complete
    # concurrent.futures.wait([future_diversity_designable, future_diversity_all])
    

    # foldseek_calculate(output_dir)
    total_time = time.time() - start_time
    with open(os.path.join(output_dir, "Metrics.txt"), "a") as f:
        f.write(f"\nTotal Evaluation time: {total_time:.3f} seconds\n")
        f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")