import numpy as np
import fakemat_experiments.bucket as bucket
import fakemat_experiments.delayline as delayline
import fakemat_experiments.delayline_symmetrical as sdelayline
import fakemat_experiments.maglattice as maglattice
import NymphESN.nymphesn as nymph
import tempESN.TempESN as temp

def create_w(mats, full_size, seed, normalise_svd=False, debug=False):
    np.random.seed(seed)
    #takes a dict of index: material, and a full size
    #create "coms matrices"
    edges = np.ones((full_size, full_size))
    offset = 0
    # for mat in mats:
    for i in range(len(mats)):
        matcoms = np.ones((full_size, full_size))
        mat = mats[i]
        incoms = np.asarray(mat.incoms)
        incoms.shape = (mat.N, 1)
        outcoms = np.asarray(mat.outcoms)
        outcoms.shape = (1, mat.N)
        incoms = np.broadcast_to(incoms, (mat.N, full_size))
        # print(incoms.shape)
        outcoms = np.broadcast_to(outcoms, (full_size, mat.N))

        incoords = tuple(np.meshgrid(np.asarray(range(full_size)), np.asarray(range(mat.N))+offset))
        # print(incoords)
        outcoords = tuple(np.meshgrid(np.asarray(range(mat.N))+offset, np.asarray(range(full_size))))
        matcoms[incoords] = incoms
        matcoms[outcoords] = outcoms
        edges = edges * matcoms
        offset+= mat.N
    #normalise svd for off-diags
    
    offset = 0
    #multiply these together with a random matrix
    W = edges * np.random.uniform(-0.5, 0.5, (full_size, full_size))
    if normalise_svd:
        s = max(abs(np.linalg.eig(W)[0])) # normalise the spectral radius of the offdiag connections to 1
        svd = np.linalg.svd(W, compute_uv=False)
        if debug:
            print(f"no normalisation: {s=}, {svd[0]=}")
        W = W / s
    if debug:
        s = max(abs(np.linalg.eig(W)[0]))
        svd = np.linalg.svd(W, compute_uv=False)
        print(f"offdiags normalised: {s=}, {svd[0]=}")
        all_coords = {}
    for i in range(len(mats)):
        mat = mats[i]
        if debug:
            matfull = mat.generate_W(seed+i, debug)
            matW = matfull[0]
            connectivity = matfull[1]
        else:
            matW = mat.generate_W(seed+i, debug)
        indices = np.asarray(range(mat.N)) + offset
        matcoords = tuple(np.meshgrid(indices, indices))
        # this needs to be set to 0 when scaling the svd
        W[matcoords] = matW
        if debug:
            edges[matcoords] = connectivity
            all_coords[i] = tuple(matcoords)
        # print(f"{matW.shape=}")
        offset+=mat.N
    if normalise_svd:
        s = max(abs(np.linalg.eig(W)[0]))
        svd = np.linalg.svd(W, compute_uv=False)
        if debug:
            print(f"total {s=}, {svd[0]=}")
            density = np.sum(edges)/(full_size**2)
            print(f"total {density=}")
        W = W / s
        if debug:
            for i in all_coords.keys():
                W_smol = W[all_coords[i]]
                s = max(abs(np.linalg.eig(W)[0]))
                svd = np.linalg.svd(W, compute_uv=False)
                print(f"{s=}, {svd[0]=}")
    # print(f"{W=}")
    return W
     

def create_esn_single_timescale(mats, full_size, seed, normalise_svd=False, K=1, debug=False):
    if debug:
        print("boop")
    W = create_w(mats, full_size, seed, normalise_svd, debug)
    esn = nymph.NymphESN(K, full_size, 1, seed=seed)
    esn.set_weights(W)
    return esn


def create_esn_multi_phase(mats, full_size, seed, normalise_svd=False, K=1):
    def rhythm_rotate(encoding, i):
        # print(encoding)
        encoding = np.array(encoding)
        value = np.packbits(encoding)
        new = np.bitwise_or(np.left_shift(value, i), np.right_shift(value, 8-i))
        encoding = np.unpackbits(new)
        # print(list(encoding))
        return list(encoding)
    # TODO: fix rhythms
    W = create_w(mats, full_size, seed, normalise_svd)
    encodings = [mats[i].encoding for i in range(len(mats))]
    rhythms = [rhythm_rotate(mats[i].rhythm, i) for i in range(len(mats))]
    #add delays (use bitwise rotate)
    
    esn = temp.Temporal_ESN(K, N=full_size, L=1, n_subreservoirs=len(mats), encodings=encodings, seed=seed, svd_dv=None)
    esn.set_weights(W)
    esn.set_rhythms(rhythms)
    return esn

rhythms_default = [[1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 0]]
def create_esn_multi_timescale(mats, full_size, seed, normalise_svd=False, K=1, rhythms=rhythms_default):
    # print("beep")
    W = create_w(mats, full_size, seed, normalise_svd)
    encodings = [mats[i].encoding for i in range(len(mats))]
    # rhythms = [[1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 0]]
    #add delays (use bitwise rotate)
    
    esn = temp.Temporal_ESN(K, N=full_size, L=1, n_subreservoirs=len(mats), encodings=encodings, seed=seed, svd_dv=None)
    esn.set_weights(W)
    esn.set_rhythms(rhythms)
    return esn

sub_size = 64
full_size = 64*3


# mats - can be used for single and multi timescales
bucket_matlist = {
    0: bucket.Bucket(sub_size, 0, 0),
    1: bucket.Bucket(sub_size, 1, 0),
    2: bucket.Bucket(sub_size, 2, 0)
}

delayline_matlist = {
    0: delayline.DelayLine(sub_size, 0, 0),
    1: delayline.DelayLine(sub_size, 1, 0),
    2: delayline.DelayLine(sub_size, 2, 0)
}

sdelayline_matlist = {
    0: sdelayline.SDelayLine(sub_size, 0, 0),
    1: sdelayline.SDelayLine(sub_size, 1, 0),
    2: sdelayline.SDelayLine(sub_size, 2, 0)
}

maglattice_matlist = {
    0: maglattice.MagLattice(sub_size, 0, 0),
    1: maglattice.MagLattice(sub_size, 1, 0),
    2: maglattice.MagLattice(sub_size, 2, 0)
}

mixed_matlist = {
    0: bucket.Bucket(sub_size, 0, 0),
    1: delayline.DelayLine(sub_size, 0, 0),
    2: maglattice.MagLattice(sub_size, 0, 0)
}

mixed_symmetrical_matlist = {
    0: bucket.Bucket(sub_size, 0, 0),
    1: sdelayline.SDelayLine(sub_size, 1, 0),
    2: maglattice.MagLattice(sub_size, 2, 0)
}
