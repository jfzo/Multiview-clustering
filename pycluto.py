import shlex, subprocess
import numpy as np
import os
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, coo_matrix
import calendar
import time
import logging

def sparse_mat_to_cluto_graph(data, outputfile, labels=None):
    sp_data = csc_matrix(data)
    N, d = sp_data.shape
    out = open(outputfile, "w")
    out.write("%d %d %d\n" % (N, d, sp_data.nnz))
    for i in range(N):
        non_zero_cols = sp_data[i, :].nonzero()[1]
        for j in non_zero_cols:
            feat = j + 1  # cluto's format starts at 1
            value = sp_data[i, j]
            out.write("%d %0.4f " % (feat, value))
        out.write("\n")
    out.close()

    if labels is not None:
        assert (len(labels) == N)
        out = open(outputfile + ".labels", "w")
        for i in range(N):
            out.write("%d\n" % labels[i])
        out.close()


def cluto_scluster(simmat, nclusters, CLUTOV_CMD="/root/cluto-2.1.2/Linux-x86_64/scluster"):

    prefix = calendar.timegm(time.gmtime())
    tempinput = "%s/%s.dat" % ("/tmp", prefix)
    # sparse_mat_to_cluto_graph(simmat, tempinput)
    np.savetxt(tempinput, simmat, fmt='%.1f', delimiter=' ', header="%d" % (simmat.shape[0]), comments='')

    # scluster -clmethod=graph -crfun=g1 -cstype=best -nnbrs=40 -grmodel=sd -nooutput -rclassfile=archivo_etiquetas archivo_grafo cantidad_grupos
    # command_order="{0} -clustfile={1}.k{2} -rclassfile={3} {1} {2}".format(CLUTOV_CMD, vectors_file, nclusters, LABEL_PATH)
    command_order = "{0} -clustfile={1}.k{2} -clmethod=graph -crfun=g1 -cstype=best -nnbrs=40 -grmodel=sd  {1} {2}".format(
        CLUTOV_CMD, tempinput, nclusters)

    # print(command_order)

    args = shlex.split(command_order)
    out = subprocess.check_output(args)
    assign_file = "{0}.k{1}".format(tempinput, nclusters)
    assignments = np.array([int(x.strip()) for x in open(assign_file)])

    # Deleting the temporal files created.
    os.remove(tempinput)
    # print("temporal file",tempinput,"deleted")
    os.remove("%s.k%d" % (tempinput, nclusters))
    # print("temporal file","%s.k%d"%(tempinput, nclusters),"deleted")

    return assignments


def cluto_vcluster(vMat, nclusters, CLUTOV_CMD="/root/cluto-2.1.2/Linux-x86_64/vcluster"):

    prefix = calendar.timegm(time.gmtime())
    tempinput = "%s/%s.dat" % ("/tmp", prefix)
    # sparse_mat_to_cluto_graph(simmat, tempinput)
    np.savetxt(tempinput, vMat, fmt='%.1f', delimiter=' ', header="%d %d" % (vMat.shape[0], vMat.shape[1]), comments='')

    # scluster -clmethod=graph -crfun=g1 -cstype=best -nnbrs=40 -grmodel=sd -nooutput -rclassfile=archivo_etiquetas archivo_grafo cantidad_grupos
    # command_order="{0} -clustfile={1}.k{2} -rclassfile={3} {1} {2}".format(CLUTOV_CMD, vectors_file, nclusters, LABEL_PATH)
    command_order = "{0} -clustfile={1}.k{2} -colmodel=none -sim=dist " \
                    "-clmethod=graph -crfun=g1 -cstype=best -nnbrs=40 " \
                    "-grmodel=sd  {1} {2}".format(CLUTOV_CMD, tempinput, nclusters)


    print(command_order)
    logging.info("Running cluto with command line {0}".format(command_order))

    args = shlex.split(command_order)
    out = subprocess.check_output(args)
    assign_file = "{0}.k{1}".format(tempinput, nclusters)
    assignments = np.array([int(x.strip()) for x in open(assign_file)])

    # Deleting the temporal files created.
    os.remove(tempinput)
    # print("temporal file",tempinput,"deleted")
    os.remove("%s.k%d" % (tempinput, nclusters))
    # print("temporal file","%s.k%d"%(tempinput, nclusters),"deleted")

    return assignments



def main():
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput

    from sklearn.datasets import make_moons
    from sklearn.metrics import euclidean_distances, f1_score
    from pycluto import cluto_scluster
    import numpy as np

    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'basic.png'

    # with PyCallGraph(output=graphviz):

    X, true_labels = make_moons(n_samples=1000, noise=.1)

    S = np.exp(-euclidean_distances(X) / (2))
    predicted = cluto_scluster(S, 2, CLUTOV_CMD='/home/juan/Documentos/cluto-2.1.2/Linux-x86_64/scluster')
    print(f1_score(true_labels, predicted, average='micro'))

    predicted = cluto_vcluster(X.data, 2, CLUTOV_CMD='/home/juan/Documentos/cluto-2.1.2/Linux-x86_64/vcluster')
    print(f1_score(true_labels, predicted, average='micro'))

if __name__ == "__main__":
    # execute only if run as a script
    main()

