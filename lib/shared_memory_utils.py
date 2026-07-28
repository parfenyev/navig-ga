import numpy as np
from multiprocessing import shared_memory
from lib.velocity_field import *

######################
#  Shared Memory
######################
def create_shared_velocity_field(flow):
    """
    Copy velocity arrays into shared memory.

    Returns
    -------
    dict
        Metadata needed by worker processes.
    """

    shm_u = shared_memory.SharedMemory(create=True, size=flow.u_data.nbytes)
    shm_v = shared_memory.SharedMemory(create=True, size=flow.v_data.nbytes)

    u_shared = np.ndarray(flow.u_data.shape, dtype=flow.u_data.dtype, buffer=shm_u.buf)
    v_shared = np.ndarray(flow.v_data.shape, dtype=flow.v_data.dtype, buffer=shm_v.buf)

    u_shared[:] = flow.u_data
    v_shared[:] = flow.v_data

    return dict(
        shm_u=shm_u,
        shm_v=shm_v,
        u_name=shm_u.name,
        v_name=shm_v.name,
        shape=flow.u_data.shape,
        dtype=flow.u_data.dtype.str,
        steady=flow.steady,
    )

def attach_shared_velocity_field(info):
    """
    Attach to an existing shared velocity field.

    Returns
    -------
    TurbFlow
    """

    shm_u = shared_memory.SharedMemory(name=info["u_name"])
    shm_v = shared_memory.SharedMemory(name=info["v_name"])
    dtype = np.dtype(info["dtype"])

    u = np.ndarray(info["shape"], dtype=dtype, buffer=shm_u.buf)
    v = np.ndarray(info["shape"], dtype=dtype, buffer=shm_v.buf)

    flow = TurbFlow(u_data=u, v_data=v, steady=info["steady"])

    # keep references alive
    flow._shm_u = shm_u
    flow._shm_v = shm_v

    return flow

def release_shared_velocity_field(shared):
    shared["shm_u"].close()
    shared["shm_v"].close()
    shared["shm_u"].unlink()
    shared["shm_v"].unlink()