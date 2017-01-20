#!/usr/bin/env python2

'''
Usage: loopfinder.py [options] FILE OUT_DIR

Options:
  --num-hashes NUM_HASHES  More hashes => better matches [default: 120].
  --min-time MIN_TIME  Minimum time to match [default: 1].
  --max-time MAX_TIME  Maximum time to match [default: 10].
  --time-distance TIME_DIST  Testing testing [default: 1].
'''

import random
from moviepy.video.fx.resize import resize
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.config import get_setting
from moviepy.tools import verbose_print, subprocess_call
import multiprocessing as mp
from collections import defaultdict
from skimage.color import rgb2gray
import numpy as np
from numpy.random import multivariate_normal
import sys
from docopt import docopt
from tqdm import tqdm
import time
import os
import subprocess as sp
from ffmpeg_writer import ffmpeg_write_video

RANDOM_SEED = 0

class LSH(object):
    def __init__(self, b):
        self.rs = None
        self.b = b

    def __call__(self, x):
        if self.rs is None:
            (dim,) = x.shape
            self.rs = multivariate_normal(np.zeros(dim), np.identity(dim), int(self.b))
        return self.rs.dot(x) >= 0


class Distance(object):
    def __init__(self):
        self.sq_norms = {}

    def __call__(self, t1, f1, t2, f2):
        def get_norm(t, f):
            if t not in self.norms:
                self.sq_norms[t] = f.dot(f)
            return self.sq_norms[t]

        u = get_norm(t1, f1)
        v = get_norm(t2, f2)
        return np.sqrt(u + v - 2 * f1.dot(f2))
        
    
def find_matches(input_file, start_time, end_time, max_d, min_d, num_hashes, progress, out_channel):
    clipFull = VideoFileClip(input_file).subclip(start_time, end_time)
    clip = clipFull.fx(resize, width=32)

    lsh = LSH(num_hashes)

    matching_frames = [] # the final result.

    hashes = {}
    in_range = []
    too_close = []
    for (t, frame) in clip.iter_frames(with_times=True, progress_bar=False):
        progress.value = (t / float(clip.duration)) * 100

        feature = rgb2gray(frame).flatten()
        
        for (i, (t2, key)) in enumerate(in_range):
            if t - t2 > max_d:
                hashes[key].remove(t2)
            else:
                in_range = in_range[i:]
                break
        
        for (i, (t2, key)) in enumerate(too_close):
            if t - t2 >= min_d:
                in_range.append((t2, key))
                hashes[key] = hashes.get(key, []) + [t2]
            else:
                too_close = too_close[i:]
                break
        
        key = tuple(lsh(feature))
        matches = [(t2, t) for t2 in hashes.get(key, [])]
        matching_frames += matches
        too_close.append((t, key))

    matching_frames = [(t1 + start_time, t2 + start_time) for (t1, t2) in matching_frames]
    out_channel.send(matching_frames)


def select_scenes(input_file, matches, min_time_span, time_distance, progress, out_channel):
    starts = {}
    for match in matches:
        (start, end) = match
        if start not in starts:
            starts[start] = match
        else:
            starts[start] = match if end > starts[start][1] else starts[start]
    
    out = []
    last_start = None
    sorted_starts = list(sorted(starts.keys()))
    i = 0
    for start in sorted_starts:
        progress.value = (i / float(len(sorted_starts))) * 100
        if (last_start is None or start > last_start + time_distance) and starts[start][1] - start > min_time_span:
            last_start = start
            out.append(starts[start])
        i += 1

    out_channel.send(out)


def write_gifs(input_file, matches, out_dir, progress):
    clip = VideoFileClip(input_file)

    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    
    for (i, (start, end)) in enumerate(matches):
        progress.value = (i / float(len(matches))) * 100
        name = "%s/%08d_%08d.mp4" % (out_dir, 100 * start, 100 * end)
        ffmpeg_write_video(clip.subclip(start, end), name, clip.fps, verbose=False, threads=1)


def divide(l, k):
    for i in range(k):
        yield l[i::k]

        
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    args = docopt(__doc__)
    input_file = args['FILE']
    out_dir = args['OUT_DIR']
    max_d = float(args['--max-time'])
    min_d = float(args['--min-time'])
    time_distance = float(args['--time-distance'])
    num_hashes = int(args['--num-hashes'])
    
    nproc = mp.cpu_count()    
    duration = VideoFileClip(input_file).duration
    
    print 'Finding loops...'
    match_procs = []
    for i in range(nproc):
        (recv, send) = mp.Pipe()
        progress = mp.Value('f')
        start_time = (duration / float(nproc)) * i
        end_time = min(duration, (duration / float(nproc)) * (i + 1) + max_d)
        p = mp.Process(target=find_matches,
                       args=(input_file, start_time, end_time, max_d, min_d,
                             num_hashes, progress, send))
        p.start()
        match_procs.append((p, recv, progress))

    matches = []
    with tqdm(total=100) as progress:
        prev_prog = 0
        while True:
            prog = int(sum(p.value for (_, _, p) in match_procs) / float(nproc))
            if prog > prev_prog:
                progress.update(prog - prev_prog)
                prev_prog = prog

            all_done = True
            procs = []
            for (proc, ch, prog) in match_procs:
                if not ch:
                    procs.append((proc, None, prog))
                elif ch.poll():
                    matches += ch.recv()
                    procs.append((proc, None, prog))
                else:
                    all_done = False
                    procs.append((proc, ch, prog))
            match_procs = procs
                    
            if all_done:
                break
            else:
                time.sleep(0.1)
        if prev_prog < 100:
            progress.update(100 - prev_prog)

    print 'Found %d loops.' % len(matches)
                        

    print 'Selecting the best loops...'
    filter_procs = []
    for i, ms in enumerate(divide(matches, nproc)):
        (recv, send) = mp.Pipe()
        progress = mp.Value('f')
        p = mp.Process(target=select_scenes,
                       args=(input_file, ms, min_d, time_distance, progress, send))
        p.start()
        filter_procs.append((p, recv, progress))

    matches = []
    with tqdm(total=100) as progress:
        prev_prog = 0
        while True:
            prog = int(sum(p.value for (_, _, p) in filter_procs) / float(nproc))
            if prog > prev_prog:
                progress.update(prog - prev_prog)
                prev_prog = prog

            all_done = True
            procs = []
            for (proc, ch, prog) in filter_procs:
                if not ch:
                    procs.append((proc, None, prog))
                elif ch.poll():
                    matches += ch.recv()
                    procs.append((proc, None, prog))
                else:
                    all_done = False
                    procs.append((proc, ch, prog))
            filter_procs = procs
                    
            if all_done:
                break
            else:
                time.sleep(0.1)
        if prev_prog < 100:
            progress.update(100 - prev_prog)
    matches = list(set(matches))

    print 'Selected the best %d loops.' % len(matches)
            

    print 'Writing gifs...'
    write_procs = []
    for i, ms in enumerate(divide(matches, nproc)):
        progress = mp.Value('f')
        p = mp.Process(target=write_gifs, args=(input_file, ms, out_dir, progress))
        p.start()
        write_procs.append((p, progress))

    with tqdm(total=100) as progress:
        prev_prog = 0
        while True:
            prog = int(sum(p.value for (_, p) in write_procs) / float(nproc))
            if prog > prev_prog:
                progress.update(prog - prev_prog)
                prev_prog = prog
            all_done = all(not proc.is_alive() for (proc, _) in write_procs)
            if all_done:
                break
            else:
                time.sleep(0.1)
        if prev_prog < 100:
            progress.update(100 - prev_prog)

if __name__ == '__main__':
    main()
