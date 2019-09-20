import numpy as np

def random_partition(num_parts, length):
    m = np.random.choice(range(1, length), size=num_parts - 1, replace=False)
    n = np.zeros(num_parts + 1, dtype='int')
    n[0] = length
    n[2:] = m

    n = sorted(n)

    diffs = []
    for i in range(num_parts):
        diffs.append(n[i + 1] - n[i])

    return diffs

def clusterStates(num_clusters, v_pi):
    sorted_states = np.argsort(v_pi)
    partitions = random_partition(num_clusters, len(v_pi))

    rep = np.zeros((len(v_pi) + 1, num_clusters))

    s = 0
    for i in range(num_clusters):
        cluster = []
        for j in range(partitions[i]):
            cluster.append(sorted_states[s])
            s += 1

        r = np.random.uniform(0, 1, size = num_clusters)
        rep[cluster, :] = (r / np.linalg.norm(r))

    return rep

