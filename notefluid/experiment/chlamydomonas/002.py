from notefile.compress import tarfile

with tarfile.open('a.gzip', "w|gz") as tar:
    tar.add('/Users/chen/data/experiment/videos/tap/150_11-3-01003.avi')
