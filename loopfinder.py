#!/usr/bin/env python2

'''
Usage: loopfinder.py [options] FILE OUT_DIR

Options:
  --num-hashes NUM_HASHES  More hashes => better matches [default: 120].
  --min-time MIN_TIME  Minimum time to match [default: 1].
  --max-time MAX_TIME  Maximum time to match [default: 10].
  --time-distance TIME_DIST  Testing testing [default: 1].
'''

from moviepy.video.fx.resize import resize
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.config import get_setting
from moviepy.tools import verbose_print, subprocess_call
import multiprocessing as mp
import cv2
from collections import defaultdict
import numpy as np
from numpy.random import multivariate_normal
import sys
from docopt import docopt
from tqdm import tqdm
import time
import os
import subprocess as sp
from ffmpeg_writer import ffmpeg_write_video

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
        
    
def select_scenes(clip, matches, min_time_span, time_distance=0):
    """
    match_thr
      The smaller, the better-looping the gifs are.
    min_time_span
      Only GIFs with a duration longer than min_time_span (in seconds)
      will be extracted.
    nomatch_thr
      If None, then it is chosen equal to match_thr
    """

    starts = {}
    for match in matches:
        (start, end) = match
        if start not in starts:
            starts[start] = match
        else:
            starts[start] = match if end > starts[start][1] else starts[start]
    
    out = []
    last_start = None
    for start in sorted(starts.keys()):
        if (last_start is None or start > last_start + time_distance) and starts[start][1] - start > min_time_span:
            last_start = start
            out.append(starts[start])
       
    return out

    
def from_clip(worker_num, clip, max_d, min_d, num_hashes, match_progress):
    """Finds all the frames tht look alike in a clip, for instance to
    make a looping gif.
    
    This returns a FramesMatches object of the all pairs of frames
    with (min_d <= t2 - t1 <= max_d).

    This is a well optimized routine and quite fast.

    Parameters
    -----------
    clip
      A MoviePy video clip, possibly transformed/resized.

    max_d
      Maximum duration (in seconds) between two matching frames.

    max_d
      Minimum duration (in seconds) between two matching frames.

    num_hashes
      Number of hashes to use for locality-sensitive hashing. More
      hashes = closer matches.

    """ 

    lsh = LSH(num_hashes)

    matching_frames = [] # the final result.

    hashes = {}
    in_range = []
    too_close = []
    for (t, frame) in clip.iter_frames(with_times=True, progress_bar=False):
        match_progress[worker_num] = (t / clip.duration) * 100

        feature = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).flatten()
        
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
                
    return matching_frames

    
def process_clip(worker_num, input_file, start_time, end_time,
                 max_d, min_d, num_hashes, time_distance, out_dir,
                 match_progress, match_done, write_progress):
    clipFull = VideoFileClip(input_file).subclip(start_time, end_time)

    clip = clipFull.fx(resize, width=32)
    matches = from_clip(worker_num, clip, max_d, min_d,
                        num_hashes, match_progress)
    matches = select_scenes(clip, matches, min_d, time_distance)
    match_done[worker_num] = 1

    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    
    for (i, (start, end)) in enumerate(matches):
        write_progress[worker_num] = (i / float(len(matches))) * 100
        name = "%s/%08d_%08d.mp4" % (out_dir, 100*start + start_time, 100*end + start_time)
        ffmpeg_write_video(clipFull.subclip(start, end), name, clip.fps, verbose=False, threads=2)

        
def main():
    args = docopt(__doc__)
    input_file = args['FILE']
    out_dir = args['OUT_DIR']
    max_d = float(args['--max-time'])
    min_d = float(args['--min-time'])
    time_distance = float(args['--time-distance'])
    num_hashes = int(args['--num-hashes'])
    
    nproc = 4
    match_progress = mp.Array('d', [0] * nproc)
    match_done = mp.Array('d', [0] * nproc)
    write_progress = mp.Array('d', [0] * nproc)

    clip = VideoFileClip(input_file)
    start_times = []
    end_times = []
    parent_pipes = []
    child_pipes = []
    for i in range(nproc):
        start_times.append((clip.duration / float(nproc)) * i)
        end_times.append((clip.duration / float(nproc)) * (i + 1) + max_d)

        (recv, send) = mp.Pipe()
        parent_pipes.append(recv)
        child_pipes.append(send)
        
    procs = []
    for i in range(nproc):
        p = mp.Process(target=process_clip,
                       args=(i, input_file, start_times[i], end_times[i],
                             max_d, min_d, num_hashes, time_distance, out_dir,
                             match_progress, match_done, write_progress))
        p.start()
        procs.append(p)

    print 'Finding matches...'
    with tqdm(total=100) as progress:
        prev_prog = 0
        while True:
            prog = int(sum(match_progress) / float(len(match_progress)))
            if prog > prev_prog:
                progress.update(prog - prev_prog)
                prev_prog = prog
            all_done = True
            for i in range(nproc):
                all_done = all_done and match_done[i] == 1
            if all_done:
                break
            else:
                time.sleep(0.1)
        if prev_prog < 100:
            progress.update(100 - prev_prog)

    print 'Writing gifs...'
    with tqdm(total=100) as progress:
        prev_prog = 0
        while True:
            prog = int(sum(write_progress) / float(len(write_progress)))
            if prog > prev_prog:
                progress.update(prog - prev_prog)
                prev_prog = prog
            all_done = True
            for i in range(nproc):
                all_done = all_done and not procs[i].is_alive()
            if all_done:
                break
        if prev_prog < 100:
            progress.update(100 - prev_prog)

if __name__ == '__main__':
    main()
