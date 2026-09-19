"""Fold aggregated gradient Fourier profiles into the figure cache.

Reads the cluster output (outputs_from_cluster/fig5_grad_out/gradient_freq.pkl, from
extract_gradient_freq.py) and writes preproc_data/gradient_freq.pkl.
"""
import os, pickle, shutil

from pnc import paths

CACHE = os.path.join(str(paths.preprocessed_data()), 'gradient_freq.pkl')
CLUSTER_OUT = os.path.join(str(paths.cluster_outputs()), 'fig5_grad_out', 'gradient_freq.pkl')


def main():
    d = pickle.load(open(CLUSTER_OUT, 'rb'))
    d['_provenance'] = 'cluster/scripts_to_cluster/fig5_gradmaps/extract_gradient_freq.py'
    pickle.dump(d, open(CACHE, 'wb'), protocol=4)
    print(f"wrote {CACHE}: {len(d['gradients'])} records, natural {d['natural_profiles_per_image'].shape}")


if __name__ == '__main__':
    main()
