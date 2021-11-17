from multiprocessing import Pool


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def run_pool(x, f, num_cores, split=False):
    if not split:
        x = split_num_segments(x, num_cores)
    print(len(x))
    # x = [[x_, i] for i, x_ in enumerate(x)]
    pool = Pool(num_cores)
    res = pool.map(f, x)
    return res


def split_num_segments(inlist, num_segments):
    res = [[] for _ in range(num_segments)]
    i = 0
    while len(inlist) > 0:
        if i < num_segments:
            res[i].append(inlist.pop())
        else:
            res[i % num_segments].append(inlist.pop())
        i += 1
    return res


def split_size_segments(inlist, seg_size):
    i = 0
    newlist = []
    while len(inlist) - len(newlist) * seg_size > seg_size:
        newlist.append(inlist[i * seg_size: (i + 1) * seg_size])
        i += 1
    if len(inlist) - len(newlist) * seg_size > 0:
        newlist.append(inlist[i * seg_size:])
    return newlist


def split_df(df, num_segments):
    idx = df.index.to_list()
    idx_segments = split_num_segments(idx, num_segments)
    return [df.iloc[idx_seg].reset_index(drop=True) for idx_seg in idx_segments]
