import bloscpack as bp
import CommonModules as CM
import torch
import scipy.sparse as sp
import numpy as np


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sp.coo_matrix(sparse_mx).astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(sparse_tensor: torch.sparse.FloatTensor):
    CoalesceTensor = sparse_tensor.cpu().coalesce()
    TensorIndices = CoalesceTensor.indices().numpy()
    TensorValues = CoalesceTensor.values().numpy()
    out = sp.coo_matrix((TensorValues, (TensorIndices[0], TensorIndices[1])), shape=sparse_tensor.shape,
                                  dtype=TensorValues.dtype)
    return out

def export_scipy_matrix(sparse_matrix, location):
    """
    :arg sparse_matrix: scipy.sparse.coo_matrix
    :arg location: output path, there will be 3 files generated for that path
    """
    if type(sparse_matrix) != sp.coo_matrix and type(sparse_matrix) != sp.csr_matrix:
        raise TypeError("unsupported sparse matrix type")
    # print(type(sparse_matrix), sparse_matrix.row.dtype, sparse_matrix.col.dtype, sparse_matrix.data.dtype)
    CM.IO.ExportToPkl(location+"_meta.pkl", sparse_matrix.shape)
    bp.pack_ndarray_to_file(sparse_matrix.row, location+'_row.blp')
    bp.pack_ndarray_to_file(sparse_matrix.col, location+'_col.blp')
    bp.pack_ndarray_to_file(sparse_matrix.data, location+'_data.blp')


def import_scipy_matrix(location, sparse_type):
    """
    Import scipy matrix as CSR or COO matrix using bloscpack efficiently.
    :arg sparse_matrix: scipy.sparse.coo_matrix
    :arg output_location: output path, there will be 3 files generated for that path
    """
    # if sparse_type not in ["coo", "csr", "bsr"]:
    if sparse_type not in ["coo", "csr"]:
        raise TypeError("unsupported sparse matrix type")

    metapath = location+"_meta.pkl"
    if CM.IO.FileExist(metapath):
        shape = CM.IO.ImportFromPkl(metapath)
    else:
        shape = None
    row  = bp.unpack_ndarray_from_file(location+'_row.blp')
    col  = bp.unpack_ndarray_from_file(location+'_col.blp')
    data = bp.unpack_ndarray_from_file(location+'_data.blp')
    # print(row.dtype, col.dtype, data.dtype)
    # sparse_matrix = scipy.sparse.coo_matrix((data, (row, col)))
    if sparse_type == "coo":
        sparse_matrix = sp.coo_matrix((data, (row, col)), shape=shape)
        # pass
    elif sparse_type == "csr":
        sparse_matrix = sp.csr_matrix((data, (row, col)), shape=shape)
        # sparse_matrix = sparse_matrix.tocsr()
    # elif sparse_type == "bsr":
    #     sparse_matrix = scipy.sparse.bsr_matrix((data, (row, col)))

    return sparse_matrix


def export_numpy_array(numpy_array, location):
    bp.pack_ndarray_to_file(numpy_array, location)


def import_numpy_array(location):
    numpy_array = bp.unpack_ndarray_from_file(location)
    return numpy_array

