# CRAC: Checkpoint-Restart Architecture for CUDA Streams and UVM

## Table of Contents

* [Introduction](#introduction)
* [TODO](#todo)
* [CITATION](#citation)
* [DMTCP](#dmtcp)

## Introduction
This is a new DMTCP(https://github.com/dmtcp/dmtcp.git) plugin to checkpoint-
restart CUDA application with noval split-process architecture. The Plugin code is in the contrib/split-cuda directory. CRAC consists of the plugin on top of DMTCP.

## TODO
We are in the process of porting our code.  It was developed for a development cluster's
specific environment.  The version here is intended to run on all recent-model NVIDIA GPUs.
We have not yet finished porting the development code.  For example, the version on the development
cluster correctly runs Gromacs, but we have not yet fixed that bug for this version.

## CITATION
The citation of this work is an accepted paper at SC'20.  The citation will be provided
here when the SC proceedings are available.  In the meantime, there is a technical
report at arxiv.org, under the title:<br>
_CRAC: Checkpoint-Restart Architecture for CUDA with Streams and UVM_<br>
by Twinkle Jain and Gene Cooperman

## DMTCP: Distributed MultiThreaded CheckPointing 
(http://dmtcp.sourceforge.net/) [![Build Status](https://travis-ci.org/dmtcp/dmtcp.png?branch=master)](https://travis-ci.org/dmtcp/dmtcp)

DMTCP is a tool to transparently checkpoint the state of multiple simultaneous
applications, including multi-threaded and distributed applications. It
operates directly on the user binary executable, without any Linux kernel
modules or other kernel modifications.

Among the applications supported by DMTCP are MPI (various implementations),
OpenMP, MATLAB, Python, Perl, R, and many programming languages and shell
scripting languages. DMTCP also supports GNU screen sessions, including
vim/cscope and emacs. With the use of TightVNC, it can also checkpoint
and restart X Window applications.  For a multilib (mixture of 32-
and 64-bit processes), see "./configure --enable-multilib".

DMTCP supports the commonly used OFED API for InfiniBand, as well as its
integration with various implementations of MPI, and resource managers
(e.g., SLURM).

To install DMTCP, see [INSTALL.md](INSTALL.md).

For an overview DMTCP, see [QUICK-START.md](QUICK-START.md).

For the license, see [COPYING](COPYING).

For more information on DMTCP, see: [http://dmtcp.sourceforge.net](http://dmtcp.sourceforge.net).

For the latest version of DMTCP (both official release and git), see:
[http://dmtcp.sourceforge.net/downloads.html](http://dmtcp.sourceforge.net/downloads.html).
