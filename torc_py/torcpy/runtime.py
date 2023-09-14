"""
(C) Copyright IBM Corporation 2019
All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

"""Implements torcpy runtime system and API."""

import copy
import ctypes
import queue
import threading
import time
import sys
import os
from termcolor import cprint
from functools import partial

# import mpi4py
# mpi4py.rc.initialize = False
# mpi4py.rc.finalize = False
import builtins
import itertools
import logging
import coloredlogs

_torc_log = logging.getLogger(__name__)
_torc_log.setLevel(logging.DEBUG)

coloredlogs.install(fmt='[%(name)s %(levelname)s]  %(message)s', stream=sys.stdout, level=logging.DEBUG)


class COMM(object):
    def __init__(self):
        self.rank = 0
        self.size = 1
    def Get_rank(self):
        return self.rank
    def Get_size(self):
        return self.size
    def barrier(self):
        pass


class myMPI(object):
    def __init__(self):
        self.COMM_WORLD = COMM()
        self.THREAD_MULTIPLE = '3'

    def Query_thread(self):
        return self.THREAD_MULTIPLE


try:
    from mpi4py import MPI
    _torc_log.debug('mpi4py module was succesfully imported')
except:
    _torc_log.warning("mpi4py could not be imported, loading a dummy MPI module.")
    MPI = myMPI()


# Constants - to be moved to constants.py
TORC_SERVER_TAG = 100
TORC_STEAL_RESPONSE_TAG = 101
TORC_TASK_YIELDTIME = 0.0001
TORC_QUEUE_LEVELS = 10

# Environment variables
TORC_STEALING_ENABLED = False
TORC_SERVER_YIELDTIME = 0.01
TORC_WORKER_YIELDTIME = 0.01

# TORC data: task queue, thread local storage, MPI communicator
torc_q = []
for _ in range(TORC_QUEUE_LEVELS+1):
    torc_q.append(queue.Queue())
torc_tls = threading.local()
torc_comm = MPI.COMM_WORLD
torc_server_thread = None
torc_deps_lock = threading.Lock()
torc_use_server = True

torc_last_qid = -1  # will need an extra lock for this
torc_executed = 0
torc_created = 0
torc_stats_lock = threading.Lock()
torc_steal_lock = threading.Lock()
torc_num_workers = int(os.getenv("TORCPY_WORKERS", 1))
torc_workers = []
torc_exit_flag = False

# Flags
_torc_shutdowned = False
_torc_launched = False
_torc_inited = False


def node_id():
    # Rank of calling MPI process
    return torc_comm.Get_rank()


def num_nodes():
    # Number of MPI processes
    return torc_comm.Get_size()


def worker_local_id():
    # Local id of current worker thread
    return torc_tls.id


def num_local_workers():
    # Number of local workers
    return torc_num_workers


def worker_id():
    # Global id of current worker thread
    return node_id() * torc_num_workers + torc_tls.id


def num_workers():
    # Total number of workers
    return num_nodes() * torc_num_workers


def _send_desc_and_data(qid, task):
    torc_comm.send(task, dest=qid, tag=TORC_SERVER_TAG)


class TaskT:
    def __init__(self, desc):
        self.desc = desc

    def input(self):
        return self.desc["args"]

    def kw_input(self):
        return self.desc["kwargs"]

    def result(self):
        return self.desc["out"]

    def __del__(self):
        if "parent" in self.desc:
            del self.desc["parent"]
        if "args" in self.desc:
            del self.desc["args"]
        if "kwargs" in self.desc:
            del self.desc["kwargs"]
        del self.desc


def submit(f, *a, qid=-1, callback=None, async_callback=True, counted=True, **kwargs):
    """Submit a task to be executed with the given arguments.

    Args:
        f: A callable to be executed as ``f(*a)``
        a: input arguments
        qid: target task queue
        callback: function to be executed on the results of ``f(*a)``
        async_callback: if True, the callback is inserted in the queue, otherwise executed immediately
        counted: if False, the task and its callback are not added to the counter of created tasks

    Returns:
        A `Future` representing the given call.
    """

    global torc_last_qid
    global torc_created

    if qid is not None and qid >= num_workers():
        _torc_log.error("submit: invalid qid value ({})".format(qid))
        raise ValueError

    # Cyclic task distribution
    if qid == -1:
        torc_stats_lock.acquire()
        qid = torc_last_qid
        torc_last_qid = qid + 1
        if torc_last_qid == num_workers():
            torc_last_qid = 0
        torc_stats_lock.release()

    if qid is not None:
        qid = qid % num_workers()
        qid = int(qid / num_local_workers())

    # Prepare the task descriptor
    task = dict()
    task["varg"] = True
    task["mytask"] = id(task)
    task["f"] = f
    task["cb"] = callback
    task["async_callback"] = async_callback
    if len(a) == 1:
        task["varg"] = False
        task["args"] = copy.copy(*a)  # firstprivate
    else:
        task["args"] = copy.copy(a)  # firstprivate
    task["kwargs"] = copy.copy(kwargs)

    # _torc_log.info("kwargs={}".format(task["kwargs"]))
    task["out"] = None  #
    task["homenode"] = node_id()
    task["deps"] = 0
    task["counted"] = counted
    task["completed"] = []
    parent = torc_tls.curr_task
    task_level = parent["level"] + 1
    if task_level >= TORC_QUEUE_LEVELS:
        task_level = TORC_QUEUE_LEVELS - 1
    task["level"] = task_level

    cb_task = None
    if callback is not None:
        cb_task = dict()
        cb_task["varg"] = False
        cb_task["mytask"] = id(cb_task)
        cb_task["f"] = callback
        cb_task["kwargs"] = {}
        cb_task["args"] = None
        cb_task["out"] = None  #
        cb_task["homenode"] = node_id()
        cb_task["deps"] = 0
        cb_task["counted"] = counted
        cb_task["cbtask"] = None
        cb_task["level"] = task_level

    if qid is not None:
        parent = torc_tls.curr_task
        task["parent"] = id(parent)
        torc_deps_lock.acquire()
        parent["deps"] += 1
        torc_deps_lock.release()

        if counted:
            torc_stats_lock.acquire()
            torc_created += 1
            torc_stats_lock.release()

        if callback is not None:
            cb_task["parent"] = id(parent)

            if counted:
                torc_stats_lock.acquire()
                torc_created += 1
                torc_stats_lock.release()

            if async_callback:
                torc_deps_lock.acquire()
                parent["deps"] += 1
                torc_deps_lock.release()
            task["cbtask"] = cb_task
        else:
            task["cbtask"] = None

    else:
        return

    # Enqueue to local or remote queue
    if node_id() == qid:
        enqueue(task_level, task)
    else:
        task["type"] = "enqueue"
        _send_desc_and_data(qid, task)

    # dictionary to object
    task_obj = TaskT(task)
    return task_obj


def _do_work(task):
    global torc_executed

    if num_nodes() > 1:
        time.sleep(TORC_TASK_YIELDTIME)

    torc_tls.curr_task = task

    f = task["f"]
    args = task["args"]
    kwargs = task["kwargs"]

    if task["varg"]:
        y = f(*args, **kwargs)
    else:
        if args is None:  # to be safe
            y = f()
        else:
            y = f(args, **kwargs)

    # send answer and results back to the homenode of the task
    if node_id() == task["homenode"]:

        task = ctypes.cast(task["mytask"], ctypes.py_object).value  # real task
        task["out"] = copy.copy(y)

        # satisfy dependencies
        parent = ctypes.cast(task["parent"], ctypes.py_object).value
        torc_deps_lock.acquire()
        parent["deps"] -= 1
        parent["completed"].append(task)
        torc_deps_lock.release()

        # trigger the callback function
        cb_task = task["cbtask"]
        if cb_task is not None:
            cb_task["args"] = TaskT(task)
            if task["async_callback"]:
                enqueue(cb_task["level"], cb_task)
            else:
                cb_task["f"](cb_task["args"])
                if task["counted"]:
                    torc_stats_lock.acquire()
                    torc_executed += 1
                    torc_stats_lock.release()

    else:
        task["out"] = y
        dest = task["homenode"]

        task["type"] = "answer"
        del task["args"]
        del task["kwargs"]
        # avoid redundant byte transfers (hint: del more)
        torc_comm.send(task, dest=dest, tag=TORC_SERVER_TAG)

    if task["counted"]:
        torc_stats_lock.acquire()
        torc_executed += 1
        torc_stats_lock.release()


def enqueue(level, task):
    """ Insert task to the queue
    """
    torc_q[level].put(task)


def dequeue():
    """Extract a task from a queue
    """
    for i in range(TORC_QUEUE_LEVELS):
        try:
            task = torc_q[i].get(True, 0)
            return task
        except queue.Empty:
            continue

    # check also the special queue for termination signal
    try:
        task = torc_q[TORC_QUEUE_LEVELS].get(True, 0)
    except queue.Empty:
        time.sleep(TORC_WORKER_YIELDTIME)
        task = {}

    return task


def dequeue_steal():
    """Extract (steal) a task from a queue
    """
    for i in range(TORC_QUEUE_LEVELS-1, -1, -1):
        try:
            task = torc_q[i].get(False)
            return task
        except queue.Empty:
            continue

    task = {}
    return task


def waitall(tasks=None, as_completed=False):
    """Suspend the calling task until all each spawned child tasks have completed
    """
    global torc_executed
    mytask = torc_tls.curr_task  # myself

    completed_tasks = None
    while True:
        torc_deps_lock.acquire()
        deps = mytask["deps"]
        torc_deps_lock.release()

        if deps == 0:
            if as_completed:
                completed_tasks = copy.copy(mytask["completed"])
            mytask["completed"] = []
            break

        while True:
            task = dequeue()
            if task is None:
                break

            if not task:
                if num_nodes() > 1 and TORC_STEALING_ENABLED:
                    task = _steal()
                    if task["type"] == "nowork":
                        time.sleep(TORC_WORKER_YIELDTIME)
                        break
                    else:
                        pass
                else:
                    break

            _do_work(task)

    torc_tls.curr_task = mytask
    return completed_tasks


def wait(tasks=None):
    # For PEP 3148
    if tasks is None:
        waitall(tasks)
    else:
        waitall(tasks)
    return tasks


def as_completed(tasks=None):
    completed_tasks = waitall(tasks, as_completed=True)
    t = []
    for c in completed_tasks:
        t.append(TaskT(c))
    return t


def _worker(w_id):
    torc_tls.id = w_id

    while True:
        while True:
            task = dequeue()
            if task is None:
                break

            if not task:
                if num_nodes() > 1 and TORC_STEALING_ENABLED:
                    task = _steal()
                    if task["type"] == "nowork":
                        time.sleep(TORC_WORKER_YIELDTIME)
                        break
                    else:
                        pass
                else:
                    break

            _do_work(task)

        if task is None:
            break


def _server():
    global torc_exit_flag, torc_executed

    torc_tls.id = -1

    status = MPI.Status()  # get MPI status object

    while True:
        while not torc_comm.Iprobe(source=MPI.ANY_SOURCE, tag=TORC_SERVER_TAG, status=status):
            time.sleep(TORC_SERVER_YIELDTIME)  # 0.01 for DIAC, 0.0001 for demo

        source_rank = status.Get_source()
        source_tag = status.Get_tag()
        task = torc_comm.recv(source=source_rank, tag=source_tag, status=status)
        source_rank = status.Get_source()
        source_tag = status.Get_tag()

        ttype = task["type"]

        if ttype == "exit":
            # Stop worker(s)
            torc_exit_flag = True
            for i in range(torc_num_workers):
                # torc_q[0].put(None)
                enqueue(TORC_QUEUE_LEVELS, None)
            break

        elif ttype == "enqueue":
            # Enqueue task

            task["source_rank"] = source_rank
            task["source_tag"] = source_tag

            # torc_q[0].put(task)
            enqueue(task["level"], task)

        elif ttype == "answer":

            real_task = ctypes.cast(task["mytask"], ctypes.py_object).value
            real_task["out"] = copy.copy(task["out"])

            # satisfy dependencies
            parent = ctypes.cast(task["parent"], ctypes.py_object).value
            torc_deps_lock.acquire()
            parent["deps"] -= 1
            parent["completed"].append(real_task)
            torc_deps_lock.release()

            # trigger the callback function
            cb_task = real_task["cbtask"]
            if cb_task is not None:
                cb_task["args"] = TaskT(real_task)
                if real_task["async_callback"]:
                    enqueue(cb_task["level"], cb_task)
                else:
                    cb_task["f"](cb_task["args"])
                    if real_task["counted"]:
                        torc_stats_lock.acquire()
                        torc_executed += 1
                        torc_stats_lock.release()

        elif ttype == "steal":
            t = dequeue_steal()
            if t is None:
                torc_q[TORC_QUEUE_LEVELS].put(t)
                t = dict()
                t["type"] = "nowork"
            elif not t:
                t["type"] = "nowork"
            else:
                t["type"] = "stolen"

            torc_comm.send(t, dest=source_rank, tag=TORC_STEAL_RESPONSE_TAG)

        else:
            _torc_log.warning("Unknown task type")


"""
Notes on the work stealing implementation:
- It is synchronous (direct response and execution of the task)
  - same tag, but it should work for multiple threads even without explicit synchronization
- Asynchronous version 1: the worker puts the stolen task in the queue
- Asynchronous version 2: the stolen task is sent to the server thread
- Better visit of remote nodes, bookkeeping of last node that had available tasks
"""


def _steal():
    global TORC_STEALING_ENABLED, torc_exit_flag

    task = None
    status = MPI.Status()  # get MPI status object
    task_req = dict()
    task_req["type"] = "steal"
    me = node_id()
    n = num_nodes()
    for i in range(0, n):
        if i == me:
            continue
        if torc_exit_flag or not TORC_STEALING_ENABLED:
            task = dict()
            task["type"] = "nowork"
            break

        # torc_steal_lock.acquire()
        torc_comm.send(task_req, dest=i, tag=TORC_SERVER_TAG)
        # OLD code, can lead to deadlock during shutdown
        # task = torc_comm.recv(source=i, tag=TORC_STEAL_RESPONSE_TAG, status=status)

        while not torc_comm.Iprobe(source=i, tag=TORC_STEAL_RESPONSE_TAG, status=status):
            time.sleep(TORC_WORKER_YIELDTIME)
            if torc_exit_flag:
                break

        if not torc_exit_flag:
            task = torc_comm.recv(source=i, tag=TORC_STEAL_RESPONSE_TAG, status=status)
        else:
            task = dict()
            task["type"] = "nowork"
            # torc_steal_lock.release()
            break

        # torc_steal_lock.release()
        if task["type"] == "nowork":
            continue
        else:
            task["source_rank"] = i
            task["source_tag"] = TORC_STEAL_RESPONSE_TAG  # not used
            break

    return task


def init():
    """Initializes the runtime library """
    global torc_server_thread, torc_use_server
    global torc_last_qid
    global torc_num_workers
    global TORC_STEALING_ENABLED
    global TORC_SERVER_YIELDTIME, TORC_WORKER_YIELDTIME
    global _torc_inited

    if _torc_inited is True:
        return
    else:
        _torc_inited = True

    # MPI.Init_thread()

    provided = MPI.Query_thread()
    if MPI.COMM_WORLD.Get_rank() == 0:
        _torc_log.warning("MPI.Query_thread returns {}".format(provided))
        if provided < MPI.THREAD_MULTIPLE:
            _torc_log.warning("Warning: MPI.Query_thread returns {} < {}".format(provided, MPI.THREAD_MULTIPLE))
        else:
            _torc_log.warning("Info: MPI.Query_thread returns MPI.THREAD_MULTIPLE")

    torc_num_workers = int(os.getenv("TORCPY_WORKERS", 1))

    flag = os.getenv("TORCPY_STEALING", "False")
    if flag == "True":
        TORC_STEALING_ENABLED = True
    else:
        TORC_STEALING_ENABLED = False

    TORC_SERVER_YIELDTIME = float(os.getenv("TORCPY_SERVER_YIELDTIME", 0.01))
    TORC_WORKER_YIELDTIME = float(os.getenv("TORCPY_WORKER_YIELDTIME", 0.01))

    torc_tls.id = 0
    main_task = dict()
    main_task["deps"] = 0
    main_task["mytask"] = id(main_task)
    main_task["parent"] = 0
    main_task["level"] = -1
    main_task["completed"] = []
    torc_tls.curr_task = main_task
    torc_last_qid = node_id() * torc_num_workers

    if num_nodes() == 1:
        torc_use_server = False

    if torc_use_server:
        torc_server_thread = threading.Thread(target=_server)
        torc_server_thread.start()


def _terminate_nodes():
    task = dict()
    task["f"] = 0
    task["type"] = "exit"

    # disable_stealing()

    nodes = num_nodes()
    for i in range(0, nodes):
        torc_comm.send(task, dest=i, tag=TORC_SERVER_TAG)


def _print_stats():
    global torc_created, torc_executed
    me = node_id()
    msg = "TORCPY: node[{}]: created={}, executed={}".format(me, torc_created, torc_executed)
    cprint(msg, "green")
    sys.stdout.flush()
    torc_created = 0
    torc_executed = 0


def finalize():
    """Shutdowns the runtime library
    """
    global torc_server_thread, torc_workers

    if node_id() == 0:

        for _ in torc_workers:
            torc_q[0].put(None)

        for t in torc_workers:
            t.join()

        if torc_use_server:
            _terminate_nodes()

    else:
        pass

    if torc_use_server:
        torc_server_thread.join()

    _print_stats()
    torc_comm.barrier()


def shutdown():
    # For PEP 3148
    global _torc_shutdowned
    if _torc_shutdowned is True:
        return
    else:
        _torc_shutdowned = True
    finalize()


def launch(main_function):
    """Launches `main_function` on worker 0 of rank 0 as the primary task of the application
    """
    global torc_workers
    global _torc_launched

    if _torc_launched is True:
        return
    else:
        _torc_launched = True

    if node_id() == 0:
        cprint("TORCPY: main starts", "green")
        sys.stdout.flush()
        torc_comm.barrier()

        torc_workers = []
        for i in range(1, torc_num_workers):
            torc_worker_thread = threading.Thread(target=_worker, args=(i,))
            torc_worker_thread.start()
            torc_workers.append(torc_worker_thread)

        torc_comm.barrier()
        if main_function is not None:
            main_function()

    else:
        torc_comm.barrier()

        torc_workers = []
        for i in range(1, torc_num_workers):
            torc_worker_thread = threading.Thread(target=_worker, args=(i,))
            torc_worker_thread.start()
            torc_workers.append(torc_worker_thread)

        torc_comm.barrier()
        _worker(0)

        for t in torc_workers:
            t.join()

        sys.stdout.flush()
        if main_function is None:
            finalize()
            sys.exit(0)


def start(main_function):
    """Initialize the library, start the primary application task and after its completion
       shutdown the library

    Args:
        main_function:

    Returns:
        Nothing
    """

    init()
    launch(main_function)
    finalize()


def gettime():
    """Returns current time in seconds. For compatibility with the C version of the library
    """
    return time.time()


def spmd(spmd_task, *arg, counted=True):
    """Submit a task to be executed with the given arguments on all MPI processes.
       Wait for completion.

    Args:
        spmd_task: A callable to be executed as ``spmd_task(*args)``
        arg: input arguments
        counted: if True, the spawned task are included in the statistics

    Returns:
        Nothing
    """

    t_all = []
    ntasks = num_nodes()
    for i in range(0, ntasks):
        task = submit(spmd_task, *arg, qid=i*num_local_workers(), counted=counted)
        t_all.append(task)
    waitall()


def _enable_stealing_task():
    global TORC_STEALING_ENABLED

    TORC_STEALING_ENABLED = True


def _disable_stealing_task():
    global TORC_STEALING_ENABLED

    TORC_STEALING_ENABLED = False


def enable_stealing():
    # Enable stealing
    spmd(_enable_stealing_task, counted=False)


def disable_stealing():
    # Disable stealing
    spmd(_disable_stealing_task, counted=False)


# map with chunksize
def _apply_chunks(f, chunk):
    return [f(*args) for args in chunk]


def _build_chunks(chunksize, iterable):
    iterable = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterable, chunksize))
        if not chunk:
            return
        yield (chunk)


def map(f, *seq, chunksize=1):
    """Return an iterator equivalent to ``map(f, *seq)``.
    Args:
        f: A callable that will take as many arguments as there are
            passed iterables.
        seq: Iterables yielding positional arguments to be passed to
                the callable.
        chunksize: The size of the chunks the iterable will be broken into
                before being passed to a worker process.

    Returns:
        An iterator equivalent to built-in ``map(func, *iterables)``
        but the calls may be evaluated out-of-order.

    Raises:
        Exception: If ``f(*args)`` raises for any values.
    """

    if chunksize == 1:
        t_all = list(builtins.map(partial(submit, f), *seq))
        waitall()

        res = []
        for task in t_all:
            res.append(task.result())

        return res

    else:
        iterable = getattr(itertools, 'izip', zip)(*seq)
        new_seq = list(_build_chunks(chunksize, iterable))
        f1 = partial(_apply_chunks, f)
        t_all = list(builtins.map(partial(submit, f1), new_seq))
        waitall()

        res = []
        for task in t_all:
            res.append(task.result())

        flat_res = [item for sublist in res for item in sublist]
        return flat_res


torc_submit = submit
torc_map = map
torc_wait = wait


class TorcPoolExecutor:
    def __init__(self):
        launch(None)

    def __enter__(self):
        return self

    @staticmethod
    def submit(f, *a, qid=-1, callback=None, async_callback=True, counted=True):
        return torc_submit(f, *a, qid=qid, callback=callback, async_callback=async_callback, counted=counted)

    @staticmethod
    def map(f, *seq, chunksize=1):
        return torc_map(f, seq, chunksize=chunksize)

    @staticmethod
    def wait(tasks=None):
        return torc_wait(tasks=tasks)

    @staticmethod
    def shutdown():
        return torc_wait(tasks=None)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return torc_wait(tasks=None)


import numpy as np


def search(arr, checkFunction, transformFunc):
    i = 0
    tasks = []
    while(i<num_workers()):
        if i < torc_num_workers:
            tasks += [submit(searchSubmitted,arr,len(arr), checkFunction, i,transformFunc)]
        else:
            tasks += [submit(searchSubmitted,None,len(arr), checkFunction, i%torc_num_workers, transformFunc)]
        i= i + 1
    waitall(tasks)
    i = 0

    while(i<torc_num_workers):
        toRet= tasks[i].result()
        if toRet != False:
            return toRet
        i= i + 1
    return False

def broadcast(value, tagId):
    size = num_nodes()
    rank = node_id()
    if rank != 0:
        return
    i= 1
    while i< size:
        torc_comm.Isend(value,dest = i,tag=101+tagId)
        i += 1

def broadcastToOtherWorkers(found_signal,tagId):
    i = 0
    while i< torc_num_workers:
        if i != tagId:
            torc_comm.Isend(found_signal,dest = 0,tag=101+i)
        i += 1



def searchSubmitted(arr, length, checkFunction, localWorkerId ,transformFunc=lambda a: a):
    size = num_nodes()
    rank = node_id()

    chunk_size = (length + size - 1) // size
    start = rank * chunk_size

    # Scatter the data to all processes
    if rank == 0:
        local_arr = torc_comm.scatter([arr[i:i+chunk_size] for i in range(0, length, chunk_size)], root=0)
    else:
        local_arr = torc_comm.scatter(arr,root=0)

    # Pad the subarray in the last rank with None values
    if rank == size - 1:
        local_arr += [None] * (chunk_size - len(local_arr))

    worker_chunk_size = (len(local_arr) + torc_num_workers - 1) // torc_num_workers

    local_arr = local_arr[localWorkerId*worker_chunk_size:(localWorkerId+1)*worker_chunk_size]

    #print("local_arr is",local_arr,"with worker",worker_id())

    # Initialize the state machine
    state = 0
    found = False
    i = 0
    index = None
    originalValue = None
    transformedValue = None

    # Shared variable to signal other processes
    found_signal = bytearray(1)
    found_signal[0] = int(found)

    while True:
        #print("state ->",state, ", i ->", i,", rank ->", rank, found)        
        if state == 0:
            # Check if any process has broadcasted the signal
            if torc_comm.Iprobe(source=MPI.ANY_SOURCE,tag=101+localWorkerId):
                torc_comm.Recv(found_signal, source=MPI.ANY_SOURCE,tag=101+localWorkerId)
            if found or found_signal[0] == 1:
                # Value was found, broadcast to stop work
                broadcast(found_signal,localWorkerId)
                break
            state = 1
        elif state == 1:
            if i < len(local_arr):
                transformedValue = transformFunc(local_arr[i])
                if checkFunction(transformedValue):
                        found = True
                        originalValue = local_arr[i]
                        index = i + start + (localWorkerId*worker_chunk_size)

                        # Broadcast the found signal to other processes
                        found_signal[0] = 1
                        broadcastToOtherWorkers(found_signal,localWorkerId)
                        if rank != 0:
                            torc_comm.Isend(found_signal,dest = 0, tag=101+localWorkerId)
                        #comm.Bcast(found_signal, root=rank)
                i += 1
                state = 0
            else:
                break

    # Gather the results to rank 0
    results = torc_comm.gather((found, index, originalValue, transformedValue), root=0)

    if rank == 0:
        # Process the results from all ranks
        for r in results:
            if r[0]:
                print("Value found at index:", r[1], "with original value", r[2], "and transformed value", r[3])
                return r
        return False

# Function to apply stencil operation
def stencilOperation(sub_data, radius, filter_func, originalShape, worker_chunk_size):
    size = num_nodes()
    rank = node_id()

    # Create buffers for exchanging data
    send_buffer_top = np.zeros(sub_data.shape[1])
    send_buffer_bottom = np.zeros(sub_data.shape[1])
    recv_buffer_top = np.zeros(sub_data.shape[1])
    recv_buffer_bottom = np.zeros(sub_data.shape[1])

    # Perform stencil computation
    if rank > 0:
        send_buffer_top = sub_data[radius:2*radius, :]
        torc_comm.Isend(send_buffer_top, dest=rank - 1,tag=worker_local_id())
        torc_comm.Recv(recv_buffer_top, source=rank - 1,tag=worker_local_id())
        sub_data[:radius, :] = recv_buffer_top

    if rank < size - 1:
        send_buffer_bottom = sub_data[-2*radius:-radius, :]
        torc_comm.Isend(send_buffer_bottom, dest=rank + 1,tag=worker_local_id())
        torc_comm.Recv(recv_buffer_bottom, source=rank + 1,tag=worker_local_id())
        sub_data[-radius:, :] = recv_buffer_bottom

    
    # Apply filter function to each index of sub_data
    result = np.zeros((sub_data.shape[0] - 2 * radius, sub_data.shape[1] - 2 * radius))
    start= radius + (worker_local_id()*worker_chunk_size)
    end = start + ((worker_local_id()+1)*worker_chunk_size)
    if end > (sub_data.shape[0] - radius):
        end = sub_data.shape[0] - radius
    for i in range(start, end):
        for j in range(radius, sub_data.shape[1] - radius):
            window = sub_data[i - radius : i + radius + 1, j - radius : j + radius + 1]
            aux = filter_func(window)
            result[i - radius, j - radius] = aux


    # Update sub_data with the computed results
    sub_data = result

    # Gather the results from all processes
    gathered_data = None
    if rank == 0:
        gathered_data = np.zeros((size*sub_data.shape[0], sub_data.shape[1]))
    torc_comm.Gather(sub_data, gathered_data, root=0)

    return gathered_data

def stencil2D(data,radius,function):
    i = 0
    tasks = []
    while(i<num_workers()):
        if i < torc_num_workers:
            tasks += [submit(stencil2DSubmited,data,radius,data.shape,function)]
        else:
            tasks += [submit(stencil2DSubmited,None,radius,data.shape,function)]
        i= i + 1
    waitall(tasks)
    i = 0

    toRet = np.zeros(data.shape)
    while(i<torc_num_workers):
        aux= tasks[i].result()
        j = aux[0]
        aux = aux[1]
        while j < data.shape[0]:
            toRet[j,:] = aux[j,:]
            j = j + torc_num_workers
        i= i + 1

    return toRet


def stencil2DSubmited(data,radius,originalShape,function):
    size = num_nodes()
    rank = node_id()
    num_rows = (originalShape[0] + size -1) // size
    #HERE    
    # Scatter the data to all processes
    if rank == 0:
        local_arr = torc_comm.scatter([data[i:i+num_rows, :] for i in range(0, size * num_rows, num_rows)], root=0)
    else:
        local_arr = torc_comm.scatter(data,root=0)

    worker_chunk_size = (num_rows + torc_num_workers - 1) // torc_num_workers

    if local_arr.shape[0] != num_rows:
        aux_arr = np.zeros((num_rows,originalShape[1]))
        aux_arr[:local_arr.shape[0],:] = local_arr
        local_arr = aux_arr

    #local_arr = local_arr[worker_local_id()*worker_chunk_size:(worker_local_id()+1)*worker_chunk_size,:]

    #print("worker",worker_local_id(),"rank",rank,"localArr",local_arr)
    sub_data = np.zeros((local_arr.shape[0] + 2 * radius, local_arr.shape[1] + 2* radius))


    if rank < size - 1:
        sub_data[radius:-radius, radius:-radius] = local_arr
    else:
        # Last rank handles the remaining rows
        sub_data[radius:-radius, radius:-radius] = local_arr

    # Perform stencil computation
    result = stencilOperation(sub_data, radius=radius, filter_func=function, originalShape=originalShape, worker_chunk_size=worker_chunk_size)

    if rank == 0:
        result = (worker_local_id(),result[:originalShape[0],:])
        return result
    

