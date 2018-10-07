
def recombine(*mappings):
    '''
    Mapping recombination between N individuals
    Input: mapping object with contigency table
    '''

    reads = mapping.sequences

    new_clusters = {0:[]}
    n_clusters = 1

    for read in reads:
        c = 0
        choice = False
        while ~choice and c<n_clusters:
            reads_c = new_clusters[c]
            # isTogether() returns the number of reads in reads_c that are in the same cluster as read
            # Needs to check if reads_c is empty
            res = [mapping.isTogether(read,reads_c) * mapping.fitness
                   for mapping in mappings]
            chooseYes = np.sum(res)
            chooseNo = [ mappings[i].fitness for i in np.where(res==0) ]
            weights = np.array([chooseNo,chooseYes])
            choice = np.random.choice([False,True],p=weights/weights.sum())
            c += 1

            if choice:
                new_clusters[c].append(read)

        if ~choice:
            n_clusters += 1
            new_clusters[n_clusters] = [read]
