#!/usr/bin/env python3
# coding: utf-8

"""
Created on Thu Sep 13 2018

Authors: 
Thomas Bonald <thomas.bonald@telecom-paristech.fr>
Nathan De Lara <nathan.delara@telecom-paristech.fr>

Weighted spectral embedding of a graph
"""

import numpy as np
from scipy import sparse, errstate, sqrt, isinf
from scipy.sparse import csgraph 
from scipy.sparse.linalg import eigsh

class SpectralEmbedding:
    """Weighted spectral embedding of a graph

        Attributes
        ----------
        embedding_ : array, shape = (n_nodes, embedding_dimension)
            Embedding matrix of the nodes

        eigenvalues_ : array, shape = (embedding_dimension)
            Smallest eigenvalues of the training matrix

        References
        ----------
        * T. Bonald, A. Hollocou, M. Lelarge. Weighted Spectral Embedding of Graphs (2018)
        * A. Ng, M. Jordan, Y. Weiss. On spectral clustering: Analysis and an algorithm (2002)
        """
    
    def __init__(self, embedding_dimension = 100, node_weights = 'uniform', eigenvalue_normalization = True):
        """
        Parameters
        ----------
        embedding_dimension : int, optional
            Dimension of the embedding space (default=100)
        eigenvalue_normalization : bool, optional 
            Whether to normalize the embedding by the pseudo-inverse square roots of laplacian eigenvalues (default=True)
        node_weights : {'uniform', 'degree', array of length n_nodes with positive entries}, optional
            Weights used for the normalization for the laplacian, W^{-1/2} L W^{-1/2}
        """
        
        self.embedding_dimension = embedding_dimension
        self.embedding_ = None
        self.eigenvalues_ = None
        self.eigenvalue_normalization = eigenvalue_normalization
        self.node_weights = node_weights

    def fit(self, adjacency_matrix, node_weights = None):
        """Fits the model from data in adjacency_matrix

        Parameters
        ----------
        adj_matrix : Scipy csr matrix or numpy ndarray
              Adjacency matrix of the graph
        node_weights : {'uniform', 'degree', array of length n_nodes with positive entries}
              Node weights 
        """
        
        if type(adjacency_matrix) == sparse.csr_matrix:
            adj_matrix = adjacency_matrix
        elif sparse.isspmatrix(adjacency_matrix) or type(adjacency_matrix) == np.ndarray:
            adj_matrix = sparse.csr_matrix(adjacency_matrix)
        else:
            raise TypeError(
                "The argument must be a NumPy array or a SciPy Sparse matrix.")
        n_nodes, m_nodes = adj_matrix.shape
        if n_nodes != m_nodes:
            raise ValueError("The adjacency matrix must be a square matrix.")
        if (adj_matrix != adj_matrix.T).nnz > 0:
            raise ValueError("The adjacency matrix must be symmetric.")
        if csgraph.connected_components(adj_matrix, directed = False)[0] > 1:
            raise ValueError("The graph must be connected.")

        # builds standard laplacian
        degrees = adj_matrix.dot(np.ones(n_nodes))
        degree_matrix = sparse.diags(degrees, format = 'csr')
        laplacian = degree_matrix - adj_matrix

        # applies normalization by node weights
        if node_weights is not None:
            self.node_weights = node_weights
        if type(self.node_weights) == str:
            if self.node_weights== 'uniform':
                weight_matrix = sparse.identity(n_nodes)
            elif self.node_weights == 'degree':
                with errstate(divide='ignore'):
                    degrees_inv_sqrt = 1.0 / sqrt(degrees)
                degrees_inv_sqrt[isinf(degrees_inv_sqrt)] = 0
                weight_matrix = sparse.diags(degrees_inv_sqrt, format = 'csr')
        else:
            if len(self.node_weights) != n_nodes:
                raise ValueError('node_weights must be an array of length n_nodes.')
            elif min(self.node_weights) < 0:
                raise ValueError('node_weights must be positive.')
            else:
                with errstate(divide='ignore'):
                    weights_inv_sqrt = 1.0 / sqrt(self.node_weights)
                weights_inv_sqrt[isinf(weights_inv_sqrt)] = 0
                weight_matrix = sparse.diags(weights_inv_sqrt, format = 'csr')
        laplacian = weight_matrix.dot(laplacian.dot(weight_matrix))
        
        # spectral decomposition
        eigenvalues, eigenvectors = eigsh(laplacian, min(self.embedding_dimension + 1, n_nodes - 1), which='SM')
        self.eigenvalues_ = eigenvalues[1:]

        self.embedding_ = np.array(weight_matrix.dot(eigenvectors[:,1:]))
        if self.eigenvalue_normalization:
            eigenvalues_inv_sqrt = 1.0 / sqrt(eigenvalues[1:])
            self.embedding_ = eigenvalues_inv_sqrt * self.embedding_

        return self

