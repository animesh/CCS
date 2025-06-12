import re

import numpy as np

aa2formula = {
    'A': {'C': 3, 'H': 7, 'N': 1, 'O': 2},
    'R': {'C': 6, 'H': 14, 'N': 4, 'O': 2},
    'N': {'C': 4, 'H': 8, 'N': 2, 'O': 3},
    'D': {'C': 4, 'H': 7, 'N': 1, 'O': 4},
    'C': {'C': 3, 'H': 7, 'N': 1, 'O': 2, 'S': 1},
    '(ac)': {'C': 2, 'H': 2, 'O': 1},
    'Q': {'C': 5, 'H': 10, 'N': 2, 'O': 3},
    'E': {'C': 5, 'H': 9, 'N': 1, 'O': 4},
    'G': {'C': 2, 'H': 5, 'N': 1, 'O': 2},
    'H': {'C': 6, 'H': 9, 'N': 3, 'O': 2},
    'I': {'C': 6, 'H': 13, 'N': 1, 'O': 2},
    'L': {'C': 6, 'H': 13, 'N': 1, 'O': 2},
    'K': {'C': 6, 'H': 14, 'N': 2, 'O': 2},
    'M': {'C': 5, 'H': 11, 'N': 1, 'O': 2, 'S': 1},
    '(ox)': {'O': 1},
    'F': {'C': 9, 'H': 11, 'N': 1, 'O': 2},
    'P': {'C': 5, 'H': 9, 'N': 1, 'O': 2},
    'S': {'C': 3, 'H': 7, 'N': 1, 'O': 3},
    'T': {'C': 4, 'H': 9, 'N': 1, 'O': 3},
    'W': {'C': 11, 'H': 12, 'N': 2, 'O': 2},
    'Y': {'C': 9, 'H': 11, 'N': 1, 'O': 3},
    'V': {'C': 5, 'H': 11, 'N': 1, 'O': 2},
    'H-': {'H': 1},
    '-OH': {'O': 1, 'H': 1}
}
atom2mass = {
    'C': 12.0,
    'H': 1.00782503223,
    'O': 15.99491461956,
    'N': 14.00307400486,
    'S': 31.97207100
}
aa2formula['C'] = {'C': 5, 'H': 10, 'N': 2, 'O': 3, 'S': 1}

def calculate_mass(seq, charge = None):
    mass = 0
    for aa in re.findall(r'\W..\W|[A-Z][a-z]?', seq):
        for atom, number_atoms in aa2formula[aa].items():
            mass += atom2mass[atom]*number_atoms
    aa_number = len(re.findall(r'[A-Z][a-z]?', seq))
    mass -= (aa_number-1)*(2*atom2mass['H'] + atom2mass['O'])
    if charge is not None:#Calculate m/z
        mass = (mass + charge)/charge
    return mass 

def decode_sequence(encoded_sequence):
    #define a mapping from amino acid to integer
    amino_acids = "_ACDEFGHIKLMNPQRSTVWY()acox"
    #decode the sequence
    reverse_mapping = dict(zip(range( len(amino_acids) + 1), amino_acids))
    decoded = [reverse_mapping[i] for i in encoded_sequence]
    #ignore the padding _
    decoded = "".join(decoded)
    decoded = decoded.replace("_", "")
    return decoded

def calc_K0_from_CCS(CCS, charge, mass):
    mass = mass + charge * 1.00727647
    k0 = CCS * np.sqrt(305 * mass * 28 / (28 + mass)) * 1/18500 * 1/charge
    return k0

def CCS_from_K0_inv(K0_inv, charge, mass):
    mass = mass + charge * 1.00727647
    CCS = K0_inv / (np.sqrt(305 * mass * 28 / (28 + mass)) * 1/18500 * 1/charge)
    return CCS

def get_norm_parameters():
    norm_params = {
        'Intensity': {'mean': 5.19186912077755, 'std': 0.7220378622351423},
        'Mass': {'mean': 3.235873169430147, 'std': 0.14154814037915567},
        'm/z': {'mean': 2.8865559391409, 'std': 0.11368319187635166},
        'Retention time': {'min': 0.0047951, 'max': 118.29}
    }
    return norm_params

def encoded_sequence(df):
    df = df.copy()
    #remove _ from the sequence
    if "Modified sequence" in df.columns:
        mod_seq_col = "Modified sequence"
    else:
        mod_seq_col = "Modified_sequence"

    df[mod_seq_col] = df[mod_seq_col].str.replace("_", "")
    #define a mapping from amino acid to integer
    amino_acids = "ACDEFGHIKLMNPQRSTVWY()acox"
    mapping = dict(zip(amino_acids, range(1, len(amino_acids) + 1)))
    # encode the sequence
    encseq = df[mod_seq_col].apply(lambda x: [mapping[i] for i in x]).values
    return encseq

def int_dataset(dat, timesteps, middle = True):
    """Returns matrix len(data) x timesteps, each entry is the aminoacid that goes in that position. Fixed Lenght = timesteps = 66.
       Shorter sequences are paded with '_' (encoded as 0)
    
    """
    empty_entry = 0
    oh_dat = (np.ones([len(dat), timesteps + 1, 1], dtype=np.int32)*empty_entry).astype(np.int32)
    cnt = 0
    for _, row in dat.iterrows():
        ie = np.array(row['encseq'])
        oe = ie.reshape(len(ie), 1)
        if middle:
            oh_dat[cnt, ((60-oe.shape[0])//2): ((60-oe.shape[0])//2)+oe.shape[0], :] = oe
        else:
            oh_dat[cnt, 0:oe.shape[0], :] = oe
        oh_dat[cnt, -1, 0] = row['Charge']
        
        cnt += 1

    return oh_dat



#numpy version of the scheduler
def calc_lr_np(step, dim_embed, warmup_steps):
    return dim_embed**(-0.5) * np.minimum(step**(-0.5), \
                                   step * warmup_steps**(-1.5))